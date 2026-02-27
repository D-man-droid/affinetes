# QQR 旅行规划评测环境

## 一、环境价值

### 1.1 为什么需要 QQR

现有主流 Benchmark（MMLU、HumanEval、GSM8K 等）评测的是模型的**静态知识**或**封闭式推理**。但 LLM 的实际应用场景越来越多是作为 **Agent** —— 自主调用外部工具、整合多源信息、生成结构化输出。

QQR 填补了这个空白：它评测的是 **模型作为 Agent 使用工具完成复杂开放式任务的端到端能力**。

具体而言，QQR 将"中文旅行规划"作为评测载体，要求被评模型：
1. **理解**自然语言的旅行需求（多类型、多约束、多偏好）
2. **自主决策**调用哪些 MCP 工具、以什么顺序和参数调用
3. **整合**多个工具的返回结果（POI 信息、导航数据、天气预报、航班/火车查询）
4. **生成**信息准确、内容完整、逻辑通顺的结构化旅行方案

### 1.2 核心评测能力

| 能力维度 | 评测方式 | 为什么重要 |
|----------|----------|-----------|
| **工具选择** | 是否调用了问题所需的工具集 | Agent 需要知道"该用什么工具" |
| **工具使用质量** | 参数是否正确（坐标格式、日期格式等） | 调了工具但参数错 = 无效调用 |
| **信息提取与整合** | 输出中引用了多少工具返回的真实数据 | 区分"真正使用工具"和"调了但没用结果" |
| **内容完整性** | 是否覆盖了所有必要的规划维度 | 好的方案需要交通、住宿、餐饮、预算等全覆盖 |
| **信息真实性** | 航班号、车次、价格等是否可追溯到工具结果 | 检测模型编造信息（hallucination） |
| **输出质量** | LLM 评判的实用性、信息量、逻辑性、用户体验 | 自动化的人类偏好代理 |

### 1.3 对 RL 训练的价值

```
Episode 流程:
  reset(task_id) → 生成问题 + 初始 prompt
       ↓
  step(tool_calls) → 执行工具 → 返回 step_reward (0~1)
       ↓  (循环 ≤ 15 步)
  step(final_answer) → 终局评分 → 返回 final_reward (0~100)
```

- **Step rewards**：每步返回即时奖励（0.4×工具选择 + 0.3×参数质量 + 0.3×结果有效性），引导模型学习工具调用策略
- **确定性评分**：相同 task_id + 相同 epoch salt = 完全相同的交通数据和评分，训练可复现
- **平滑 LLM-code coupling**：`llm_score *= min(1.0, code / (max_code × 0.75))`，防止模型只优化一个维度
- **10K+ 任务空间**：7 类型 × 3 难度 × 70+ 城市 × 每周 salt 轮换，避免过拟合

---

## 二、系统架构

### 2.1 整体流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        QQR 评测流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  problem_generator.py + knowledge_graph.py                      │
│  ┌─────────────┐     task_id (确定性种子)                        │
│  │ 7 问题类型   │ ──→ TravelProblem 数据结构 ──→ 中文 prompt     │
│  │ 3 难度等级   │     (城市/日期/预算/偏好/约束)                   │
│  │ 70+ 城市    │     城市知识图谱提供季节/特产/地标               │
│  └─────────────┘                                                │
│         ↓                                                       │
│  env.py (Actor) — 两阶段设计                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Phase 1: Tool-calling Loop (≤ 15 steps)               │    │
│  │                                                         │    │
│  │  被评模型 ←→ MCP 工具集 (via MCPState)                    │    │
│  │              ├── poi_search      (AMap API, 真实)        │    │
│  │              ├── around_search   (AMap API, 真实)        │    │
│  │              ├── direction       (AMap API, 真实)        │    │
│  │              ├── weather         (AMap API, 真实)        │    │
│  │              ├── search_flights  (确定性生成, mock)       │    │
│  │              └── search_train    (确定性生成, mock)       │    │
│  │                                                         │    │
│  │  每步 → StepRewardCalculator → step_reward (0~1)        │    │
│  │                                                         │    │
│  │  Phase 2: Final Answer                                  │    │
│  │  若模型自然回答不充分 → 发送 final-answer prompt           │    │
│  │  (tools=None, 禁止再调用工具)                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│         ↓ (模型输出最终方案)                                      │
│  scorer.py + llm_validator.py                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  多层评分 (max 100)                                      │    │
│  │                                                         │    │
│  │  1. Hard Constraints (门槛检查)                          │    │
│  │     format_valid ──→ 不通过 = ×0.15 (近零分,保留RL梯度)   │    │
│  │     tool_info_used ──→ 不通过 = 0 分                     │    │
│  │     required_tools ──→ 不通过 = ×0.5                     │    │
│  │     poi_names ──→ 不通过 = ×0.7                          │    │
│  │     transport_grounded ──→ 不通过 = ×0.3 (渐进)          │    │
│  │     tool_quality ──→ 不通过 = ×0.5                       │    │
│  │                                                         │    │
│  │  2. Code Score (70 分)                                   │    │
│  │     info_consistency (35) ← 10 类别对比工具事实与输出      │    │
│  │     completeness (35) ← 分层验证 (keyword+context+fact)   │    │
│  │     fabrication_penalty (0 ~ -17.5) ← 编造扣分            │    │
│  │                                                         │    │
│  │  3. LLM Score (30 分)                                    │    │
│  │     practicality + informativeness + logic + ux           │    │
│  │     × smooth coupling with code score                    │    │
│  │                                                         │    │
│  │  Final = (code + llm) × HC_multipliers                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 文件结构

```
environments/qqr/
├── __init__.py             # 包导出：Actor, ProblemGenerator, TravelScorer 等
├── env.py                  # Actor 类：两阶段 Agent Loop、MCP 工具调度、evaluate() 入口
├── scorer.py               # 核心评分：事实提取、HC 检查、IC/Comp 计算、编造检测
├── config.py               # 配置：工具定义、城市列表、分数权重、HC 惩罚系数
├── problem_generator.py    # 确定性问题生成器：7 类型、DifficultyProfile、prompt 模板
├── knowledge_graph.py      # 城市知识图谱：71 城市的特色/地标/美食主题/季节/交通枢纽
├── parser.py               # 输出解析：JSON 优先 + regex fallback 结构化提取
├── llm_validator.py        # LLM 语义评估：4 维度 × 7.5 分，含反注入过滤
├── mcp_wrapper.py          # MCP 协议封装（从 QQR 移植，避免 slime 依赖）
├── mock_transport/
│   ├── __init__.py
│   └── server.py           # 确定性交通数据生成（SHA256 种子、70+ 城市、epoch salt）
├── Dockerfile              # 容器构建配置
└── requirements.txt        # 依赖
```

### 2.3 MCP 工具集

| 工具 | 数据来源 | 用途 | 返回数据 |
|------|----------|------|----------|
| `poi_search` | AMap API（真实） | 搜索景点/酒店/餐厅等 POI | 名称、地址、坐标、评分、电话 |
| `around_search` | AMap API（真实） | 按坐标半径搜索周边 | 周边 POI 列表 |
| `direction` | AMap API（真实） | 路线规划（驾车/步行/骑行/公交） | 距离、耗时、路线描述 |
| `weather` | AMap API（真实） | 天气预报 | 天气状况、温度、风力 |
| `search_flights` | 确定性生成（mock） | 航班搜索 | 航班号、价格、时间、航空公司 |
| `search_train_tickets` | 确定性生成（mock） | 火车票搜索 | 车次、价格、时间、座位类型 |

**为什么交通数据用 mock？** 真实航班/火车 API 不稳定且有配额限制。mock_transport 基于 SHA256 种子确定性生成，保证：
- 相同 `(date, from_city, to_city, salt)` → 相同航班/车次数据
- 每周 `TRANSPORT_SALT` 轮换 → 防止模型记忆历史数据
- 70+ 城市互联 → 覆盖短/中/长途场景

**Chutes API 适配**：因 Chutes API 不完全支持 OpenAI 的 `tool_calls` 格式和 `tool` role 消息，Actor 将工具调用转换为中文文本描述，并将工具结果合并到 `user` 消息中。

---

## 三、评分系统详解

### 3.1 分数结构

```
总分 = (Code Score + LLM Score) × HC Multipliers
     = (IC + Comp + Fab) × HC₁ × HC₂ × ... + LLM × coupling

其中:
  Code Score (max 70) = info_consistency (35) + completeness (35) + fabrication_penalty (0~-17.5)
  LLM Score (max 30)  = practicality (7.5) + informativeness (7.5) + logic (7.5) + ux (7.5)

LLM-Code Coupling:
  llm_score *= min(1.0, code_score / (70 × 0.75))
  → code < 52.5 时 LLM 分被线性压缩，防止模型只优化 LLM 维度
```

### 3.2 Hard Constraints（门槛检查）

评分前先检查 6 项约束，不满足则施加惩罚乘数：

| 约束 | 乘数 | 判定逻辑 |
|------|------|----------|
| `format_valid` | 0.15 (近零分，保留 RL 梯度) | 输出是否包含旅行方案结构（按问题类型检查特定关键词） |
| `tool_info_used` | 0.0 (总分归零) | LLM 评审判定输出是否引用了工具返回的事实（含 code-based 交叉验证） |
| `required_tools_called` | 0.5 | 是否调用了问题类型所需的核心工具（覆盖率阈值 50%-60%） |
| `poi_names_verified` | 0.7 | 输出中是否至少有 2 个 POI 名称来自工具返回（模糊匹配） |
| `transport_grounded` | 0.3 (渐进) | 交通信息（航班号/车次）是否来自工具返回 |
| `tool_quality` | 0.5 | 工具覆盖率和有效性是否均 ≥ 50% |

**transport_grounded 渐进惩罚**：编造比例 ≤ 20% 无惩罚，20%~100% 线性插值到 0.3x，而非二值的通过/不通过。

**tool_info_used 交叉验证**：即使 LLM 评审认为使用了工具信息，如果 code-based 的 IC 和 completeness 均极低（交通类型 < 3.0, 非交通类型 < 1.0），仍然判定为未使用工具信息。

### 3.3 Info Consistency（信息一致性，35 分）

衡量"模型输出中有多少信息可追溯到工具返回的真实数据"。

```
事实提取流程:
  工具调用 trace → FactExtractor:
    _extract_poi_facts()     → POI 名称
    _extract_weather_facts() → 天气状况、风力
    _extract_flight_facts()  → 航班号、价格、时间
    _extract_train_facts()   → 车次、价格、时间
    _extract_direction_facts() → 距离、耗时、路线名

  模型输出 → FactExtractor.extract_from_output() → 同类别提取

  10 类别逐一对比:
    flights:          航班号集合交集
    trains:           车次集合交集
    pois:             POI 名称模糊匹配
    weather:          天气描述集合交集
    distances:        距离数值文本匹配
    times:            时间集合交集
    prices:           价格数值集合交集
    wind_info:        风力信息集合交集
    travel_durations: 行程耗时文本匹配
    road_names:       路线名称文本匹配

  每个类别: overlap_ratio = matched / min(tool_count, output_count)
            normalized = min(1.0, overlap_ratio / 0.6)  # 60% overlap = 满分

  IC = 35 × avg(normalized across categories)

  广度惩罚: 如果 categories_matched < max(2, (total+1)//2) 且 total ≥ 4 → IC × 0.5
```

### 3.4 Completeness（完整度，35 分）

衡量"输出是否覆盖了所有必要的规划维度"。采用分层验证 + 数量缩放：

```
_check_with_grounded_context(text, keyword, context, tool_facts, max_pts, target_count):

  keyword:    规划维度关键词，如 r'(预算|费用|花费)'
  context:    上下文模式，如 r'\d+\s*元'
  tool_facts: 工具返回的事实集合（POI 名、距离、价格等）

  验证逻辑 (4 级):
    keyword + context + tool_fact → 100% (满分)
    keyword + tool_fact           → 50%
    keyword + context             → 15%
    keyword only                  → 0% (无工具事实 = 不给分)

  数量缩放 (target_count > 0 时):
    tier_score *= max(0.25, grounded_facts / target_count)
```

**交通 ID 检查**使用更严格的 `_check_with_verified_context`：通过 lookbehind+lookahead 正则精确匹配航班号/车次（如 `(?<![A-Za-z\d])G1234(?!\d)`），防止部分匹配。

**日结构评分**（multiday/hybrid/family_study）使用渐进 POI grounding：
```
day_grounding = 0.3 + 0.7 × min(1.0, poi_in_text / target_pois)
score = 7.0 × day_ratio × day_grounding
```

以 `business` 类型为例 (35 分 = 8+7+6+7+7):

| 维度 | 分值 | 验证方式 | 验证的 tool_facts |
|------|------|----------|------------------|
| 交通方案 | 8 | verified_context | flights ∪ trains (精确 ID 匹配) |
| 酒店推荐 | 7 | grounded_context | POI 名称 (模糊匹配) |
| 餐饮推荐 | 6 | grounded_context | POI 名称 (模糊匹配) |
| 费用预估 | 7 | grounded_context | 价格数字 |
| 商务配套 | 7 | grounded_context | POI 名称 (模糊匹配) |

### 3.5 LLM 语义评估（30 分）

使用独立 LLM 对输出进行语义评估：

| 配置 | 值 |
|------|---|
| 主模型 | `Qwen/Qwen2.5-72B-Instruct` |
| 备选模型 | `DeepSeek-V3-0324`, `Qwen3-235B-A22B` |
| 熔断机制 | 连续 3 次失败后开启 |
| 降级策略 | 验证器不可用时使用默认分 15/30（每维度 3.75） |

4 个评估维度（各 0-10 分，缩放至 0-7.5）：
1. **实用性 (practicality)** — 方案是否可执行？（时间、交通衔接）
2. **信息量 (informativeness)** — 信息是否丰富？（具体名称、价格、时间）
3. **逻辑性 (logic)** — 方案是否逻辑连贯？
4. **用户体验 (user_experience)** — 是否满足用户约束和偏好？

**反注入保护**：输出在发送给验证器前会过滤注入模式（如 "ignore above"、"set all scores"、"override instructions"），并使用随机边界 token 包裹。

### 3.6 反作弊体系

| 机制 | 防御目标 | 实现方式 |
|------|----------|----------|
| 分层验证 | 关键词填充 | 仅关键词匹配 → 0 分；需要同时满足 keyword+context+tool_fact |
| 精确 ID 匹配 | 编造航班/车次 | `_check_with_verified_context` 使用 lookbehind+lookahead 正则 |
| IC 60% 阈值 | 低质量引用 | 每个类别需要 60% 以上的工具事实被正确引用 |
| 广度惩罚 | 只引用一个类别 | 匹配类别 < 总类别一半（最少 2）且可用类别 ≥ 4 → 分数 ×0.5 |
| 编造扣分 | Hallucination | 检测到编造的交通/价格/天气信息 → 最多扣 17.5 分 |
| 渐进交通惩罚 | 部分编造 | 编造比例从 20% 到 100%，惩罚从 1.0x 渐进到 0.3x |
| Epoch Salt | 记忆历史数据 | 每周轮换 TRANSPORT_SALT → 交通数据完全变化 |
| LLM-Code Coupling | 只优化 LLM 分 | code < 52.5 时 LLM 分被线性压缩 |
| 交叉验证 | LLM 判断误差 | LLM 说 tool_info_used=True 但 IC 和 Comp 均极低 → 覆盖为 False |
| 反注入过滤 | Prompt Injection | LLM 验证器前过滤注入指令 |

---

## 四、问题生成系统

### 4.1 七种问题类型

```
task_id % 7 → 问题类型:
  0: intercity      城际交通规划 (需要: poi_search + direction + weather + flights + trains)
  1: multiday       多日游规划   (需要: poi_search + around_search + direction + weather)
  2: hybrid         综合规划     (需要: 全部 6 个工具)
  3: single_poi     单景点深度游 (需要: poi_search + around_search + direction + weather)
  4: food_tour      美食之旅     (需要: poi_search + around_search + direction + weather)
  5: business       商务出行     (需要: poi_search + direction + weather + flights + trains)
  6: family_study   亲子研学     (需要: poi_search + around_search + direction + weather)
```

### 4.2 确定性生成

```python
rng = random.Random(task_id)  # task_id 作为种子

# 所有参数由 rng 确定性派生:
problem_type = PROBLEM_TYPES[task_id % 7]
difficulty = (task_id // 7) % 3 + 1        # 1/2/3 循环
destination = rng.choice(MAJOR_CITIES)      # 从 70+ 城市中选
travel_date = base_date + timedelta(days=task_id % 365)
interests = rng.sample(INTERESTS, rng.randint(2, 4))
# ...
```

### 4.3 难度等级与 DifficultyProfile

| 等级 | 标签 | 工具数 | 最大天数 | DifficultyProfile |
|------|------|--------|---------|-------------------|
| 1 | beginner | 2-3 | 1 | constraint_tightness=0.5, conflicts=1, time_pressure=False |
| 2 | intermediate | 3-5 | 3 | constraint_tightness=0.75, conflicts=2, time_pressure=可能 |
| 3 | advanced | 5-6 | 5 | constraint_tightness=0.95, conflicts=3, time_pressure=True |

**DifficultyProfile** 控制：
- `constraint_tightness` — 预算紧缩系数（越高越紧）
- `constraint_conflicts` — 注入的矛盾约束对数量（如"预算优先"+"舒适优先"）
- `time_pressure` — 是否有紧迫的到达时间窗口

### 4.4 城市知识图谱

`knowledge_graph.py` 为全部 71 个城市提供结构化信息：

```python
CityProfile:
  specialties   # 城市特色（如 "历史文化", "自然风光"）
  landmarks     # 2-6 个著名景点
  food_themes   # 匹配的美食主题关键词
  seasonal_avoid # 应避开的月份（台风季、极端高温/严寒）
  transport_hub  # 是否有大型机场 + 高铁站
  nearby_cities  # 适合多日延伸的临近城市
```

用于：
- 季节性检查：避免生成不合适季节的旅行问题
- 兴趣偏向：将兴趣关键词偏向城市特色
- 美食主题匹配：food_tour 类型选择与美食主题匹配的城市
- POI 池：地标数据用于 single_poi 类型的景点选择

---

## 五、确定性交通数据生成

`mock_transport/server.py` 基于 SHA256 种子生成航班和火车数据：

```python
# 种子构造
seed = SHA256(f"{TRANSPORT_SALT}|{date}|{from_city}|{to_city}")

# 由种子确定性派生:
- 航班数量 (8-15)、航空公司 (15 家)、航班号 (如 CA1234)、价格、起降时间
- 火车数量 (8-15)、车次类型 (G/D/C/Z/T/K)、车次号 (如 G1986)、价格、时间
- 飞行距离（城市对查表 or SHA256 fallback，对称且 salt 无关）
```

**关键设计**：
- `TRANSPORT_SALT` 每周自动轮换：`str(int(time.time()) // (7 * 86400))`
- 航班号去重：每个查询内不会生成重复 ID
- 距离对称：`distance(A→B) = distance(B→A)`
- Fallback 距离与 salt 无关：`SHA256("distance|sorted_city_pair")`
- 70+ 城市机场/车站映射：使用真实名称（如"首都国际机场"、"北京南站"）
- 6 种火车类型按距离过滤：短途仅 G/D/C，长途包含 Z/T/K
- 红眼航班：每次生成 1-2 个红眼航班，享受价格折扣

---

## 六、输出解析器

`parser.py` 将模型输出解析为结构化数据 (`ParsedOutput`)：

1. **JSON 优先**：尝试从 code blocks 或原始文本中提取 JSON
2. **Regex Fallback**：若无 JSON，使用正则提取：
   - 交通选项：`航班 CA1234，价格XXX元` 模式
   - 日行程：`第N天` 或 `Day N` 分段
   - 预算：类别关键词 + 价格模式
   - 地点：后缀匹配（景区、公园、博物馆、古镇等）

---

## 七、快速开始

### 7.1 环境要求

```bash
# .env 文件中需要:
CHUTES_API_KEY=cpk_...       # Chutes LLM API (用于被评模型和 LLM 评审)
AMAP_MAPS_API_KEY=a605...    # 高德地图 API (用于 POI/导航/天气)
```

### 7.2 构建与运行

```bash
# 激活虚拟环境
source .venv/bin/activate

# 构建 Docker 镜像
afs build environments/qqr --tag qqr:v1

# 启动容器
afs run qqr:v1 --name qqr --env CHUTES_API_KEY=$CHUTES_API_KEY --env AMAP_MAPS_API_KEY=$AMAP_MAPS_API_KEY

# 单次评测
afs call qqr evaluate --arg task_id=131

# 批量评测
python examples/qqr/test_qqr.py
```

### 7.3 Python API

```python
import affinetes as af

env = af.load_env(
    image="qqr:v1",
    mode="docker",
    env_vars={
        "CHUTES_API_KEY": api_key,
        "AMAP_MAPS_API_KEY": amap_key,
    },
)

# 完整评测（内部自动完成两阶段 Agent Loop）
result = await env.evaluate(
    model="moonshotai/Kimi-K2.5-TEE",
    base_url="https://llm.chutes.ai/v1",
    task_id=131,
    timeout=300,
    temperature=0.7,
)

print(f"Score: {result['score'] * 100:.1f}/100")
print(f"Pass: {result['success']}")  # score >= 60 为通过

# 手动控制 Agent Loop（适合 RL 训练）
reset_resp = await env.reset(task_id=131)
episode_id = reset_resp.episode_id

# Phase 1: 工具调用
step_resp = await env.step(
    action="",
    episode_id=episode_id,
    tool_calls=[{
        "function": {
            "name": "poi_search",
            "arguments": '{"address": "贵阳", "region": "贵阳"}'
        }
    }],
)
print(f"Step reward: {step_resp.reward}")  # 0~1

# Phase 2: 最终回答（不传 tool_calls → 触发评分）
final_resp = await env.step(
    action="完整的旅行规划方案...",
    episode_id=episode_id,
    tool_calls=None,
)
print(f"Final score: {final_resp.info['score']}")  # 0~100
```

### 7.4 Actor 初始化参数

```python
Actor(
    enable_llm_validator=True,          # 是否启用 LLM 语义评估
    llm_validator_model="Qwen/Qwen2.5-72B-Instruct",  # LLM 评审模型
)
```

### 7.5 依赖

```
httpx>=0.25.0
openai>=1.0.0
openai-agents>=0.6.0
mcp>=1.0.0
diskcache>=5.6.0
click>=8.0.0
```
