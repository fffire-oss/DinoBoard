# 审计 follow-up（2026-05-13）

来源：`Desktop/dinoboard-claims-audit.md`，复核 commit `e376eaf`。
我把所有我同意的问题列在这里，按改动难度从易到难排序。原报告条目编号在每项末尾用 `[#N]` 标注。

总体判断：DEC-003（删 session 每步 randomize_unseen）+ step_count wrapper 这次架构收紧本身落地干净，但 README / FEATURES_OVERVIEW / ALGORITHM_OVERVIEW / GAME_DEVELOPMENT_GUIDE / 部分头注释 + 测试 docstring 没有同步更新，外加几处 README 过满表述。绝大多数是文档债，少数是接口债。

---

## A. 极易（单行 / 单段落改动，半小时内）

### A1. `CONFIG_REFERENCE.md` 引用不存在的 `_BASE` [#13]

`docs/guide/CONFIG_REFERENCE.md` 写"BASE 默认值见 `training/mcts_profile.py:_BASE`"，实际变量名是 `_BASE_DICT`。直接改锚点。

### A2. `FEATURES_OVERVIEW.md` ISMCTS 锚点错位 [#9]

链接到 `ALGORITHM_OVERVIEW.md#8-mcts--ismcts-over-a-dag`，但 §8 是 belief / randomize_unseen，§9 才是 MCTS = ISMCTS over DAG。修锚点。

### A3. `ALGORITHM_OVERVIEW.md` §7 "tracker 不参与决策路径" 错误 [#5]

原文："tracker 不进 hash、不进协议、不参与决策路径——只在 encoder 提取特征时被读取"。
事实：`randomize_unseen` 在每个 MCTS sim 入口被调用（`engine/search/net_mcts.cpp`），决定该 sim 的世界、合法动作、value backup。

正确说法：tracker 不进 DAG hash / 不进 wire protocol；但 `randomize_unseen` 是 ISMCTS 决策管线必经一步，encoder 读 tracker 是另一条用途。

### A4. `GAME_DEVELOPMENT_GUIDE.md` §3.3 仍写"开发者要更新 step_count_" [#8]

DEC-003 + 现在的 `IGameRules` wrapper 已经把 step_count_ 字段藏成 protected，作者既看不到也无法忘记。但 §3.3 仍写"开发者只需实现 `do_action_fast` / `undo_action` 时正确更新 `step_count_` 和 RNG 状态"。删掉这句 / 改成"step_count_ 由框架 wrapper 自动 ±1，作者无需也无法直接维护"。

### A5. `tests/framework/test_selfplay_no_truth_in_ai_path.py` docstring 与代码不一致 [#17]

docstring 写 "the four §A.a in-scope games"、Love Letter intentionally out、session hidden 每 ply 重采，但实际 `IN_SCOPE_GAMES = ["tictactoe", "quoridor", "azul", "splendor", "loveletter"]`，DEC-003 也已删 session freshening。改 docstring 与代码 + 当前契约对齐。

### A6. `platform/ai_service/routes.py` 头注释引用旧配置源 [#12]

注释写 `difficulty_overrides.expert` + 硬编码 fallback，实际 `sessions.py` 已改用 `resolve_profile(base, "web_expert")` 读六个命名 profile。改头注释。

### A7. README "all training hyperparameters live in `games/<game>/config/game.json`" 过满 [#16]

事实：selfplay / arena / eval 三个 profile 在 `game.json`，web_expert / web_casual / analysis 在 `web.json`。
改成 "training loop hyperparameters in game.json; MCTS profiles split between game.json (selfplay/arena/eval) and web.json (web_expert/web_casual/analysis)" 之类。

### A8. README "a single set of CI tests covers everyone" 过满 [#3]

事实：framework matrix carrier `FRAMEWORK_GAMES = ["quoridor", "azul", "loveletter"]`；只有部分结构性测试用 `enabled_games()` 自动扩展（比如新加的 `test_step_count_strict_increase`），per-game checklist 仍要每个游戏自己写。
改成"框架核心 invariants 用少数 carrier 游戏覆盖，per-game 行为靠 `tests/<game>/test_checklist.py` 单独维护；新游戏不会自动进入所有框架测试"。

---

## B. 中等（多文件 sweep / 决策点）

### B1. README + FEATURES_OVERVIEW "descent fully deterministic" 过满 [#1]

`net_mcts.cpp` 注释自己已经承认 `sim_rng` 同时驱动 root determinization 和 descent 的 `do_action_fast`（Azul 每轮工厂 refill 在 descent 里也消费 sim_rng）。
改成：每个 sim 持独立 RNG；root 先按 tracker 采样隐藏信息，descent 期间该 RNG 仍可能被 `do_action_fast` 消费处理物理随机。`ALGORITHM_OVERVIEW.md` §3 RNG 表已经写对了，把高层文档对齐到低层即可。

### B2. README `test_api_mcts_policy_invariance` 背书过头 [#2]

测试参数 `LEAK_SENSITIVE_GAMES = ["loveletter"]`，Splendor 因 replay / `self_reserve_deck` 已知问题主动排除，Web 路径不直接覆盖；阈值还允许 argmax mismatch ≤ 0.65、TV ≤ 0.40。
README 那句"selfplay / web / API are behaviorally equivalent at the MCTS level (guarded by ...)"改成更准确的范围声明，或者把 Splendor 重新接入测试再背书。

### B3. `randomize_unseen` 签名缺 `observer` 的旧示例 [#7]

`engine/core/belief_tracker.h` 真签名是 `randomize_unseen(IGameState&, int observer, std::mt19937_64&)`。
ALGORITHM_OVERVIEW.md / GAME_DEVELOPMENT_GUIDE.md / CLAUDE.md / VIBE_CODING_ONBOARDING.md 中仍残留缺 `observer` 参数的示例片段。grep 改全。`observer` 不是装饰，丢了示例就是教坏后来人。

### B4. `KNOWN_ISSUES.md` 旧架构状态没标"历史归档" [#11]

旧 BUG-008 / OB-005 末尾仍写"selfplay_runner 改成持有 N 个 per-perspective tracker ... MCTS root 暂时仍走 legacy 单 tracker 路径"，BUG-028 / BG-008 长文还保留 "apply_observation 末尾统一调 randomize_unseen" 的 MVP-B 叙述。
当前代码：`selfplay_runner` MCTS 已用 `per_perspective_trackers[player]`；DEC-003 已删 session freshening。
做法：在这些段落顶端加一行"⚠️ 历史归档，当前架构以 DEC-003 + 当前代码为准"，避免读者误判修复进度。

### B5. `GAME_DEVELOPMENT_GUIDE.md` §7 残留旧 JSON 配置结构 [#14]

§7 列出 `network / selfplay / replay / eval / arena / optimizer / tail_solve / mcts / mcts_schedule / ...`，但 `CONFIG_REFERENCE.md` 现在围绕 `mcts_profiles.{selfplay,arena,eval}` + `mcts_profiles.{web_expert,web_casual,analysis}` 组织，独立 `mcts_schedule` / 顶层 `mcts` 已不是当前结构。
做法：§7 的列表换成 `mcts_profiles.*` 体系，删掉 `mcts_schedule` 等已废弃顶层段。

### B6. DEC-003 后 session-side `randomize_unseen` 旧描述全仓清扫 [#6]

最大的一笔文档债。DEC-003 已删 session 每步 freshening，但以下位置仍在讲旧管线：

- `ALGORITHM_OVERVIEW.md` §1.1（hidden 字段被 tracker 重采）
- `ALGORITHM_OVERVIEW.md` §5.4 链路图（apply_observation 画成 hidden 由 tracker 重采）
- `ALGORITHM_OVERVIEW.md` §8.1（调用语义包括 apply_observation 末尾）
- `ALGORITHM_OVERVIEW.md` §9.8 伪代码（`per_perspective_trackers[p]->randomize_unseen(seat[p], p, freshen_rng[p])`，且与同节后文自相矛盾）
- `FEATURES_OVERVIEW.md`（仍写 session hidden 每步重采）
- `docs/guide/GAME_DEVELOPMENT_GUIDE.md` §10.4（randomize_unseen 两个调用点，第二个写成 observe 末尾）
- `engine/core/belief_tracker.h` 注释（"called at the end of apply_observation"）
- `engine/runtime/selfplay_runner.cpp` 注释（"freshening of hidden happens after observe"）

策略：grep `randomize_unseen` + `apply_observation`，逐处对照 DEC-003 改写。规则统一为"`randomize_unseen` 唯一调用点是 `engine/search/net_mcts.cpp` sim 入口的 sim_tracker 克隆；session 状态上的 viz=0 槽位永远不被框架 freshen"。

### B7. Azul 在 tracker / manifest hidden_info 的口径上自相矛盾 [#4]

事实：`games/azul/azul_register.cpp` 注册 tracker，但 `randomize_unseen` no-op（bag 改成 per-color counts，没东西可采样）。`games/manifest.json` 的 Azul entry 带 `["hidden_info", "snapshot", "tracker"]`，`_capabilities_doc` 里 `hidden_info` 的定义是"perspective-private slots (viz != all_public)"——Azul 没有这种槽位。
但 `ALGORITHM_OVERVIEW.md` 某些章节又把 Azul 归成"不注册 tracker"，`GAME_DEVELOPMENT_GUIDE.md` §10 也写 Azul 不必注册。
`docs/guide/AI_API.md` + `platform/ai_service/routes.py` 注释又把 Azul 归进"有隐藏信息的游戏"。

需要决策：
- 选项 A：Azul 真的应该是"完全公开 + 物理随机"，那 manifest 去掉 `hidden_info` 标记、tracker 也不该注册（或保留为 public-aggregates 缓存的合法用途，加 capability 区分）。
- 选项 B：保留 tracker 但在 manifest `_capabilities_doc` 加第三种 capability（例如 `public_aggregate_cache` 或 `physical_random`），文档区分"非对称隐藏 / 公开物理随机 / message-driven snapshot"三类，不再用 `hidden_info` 包打天下。

不论选哪条，都要把 ALGORITHM_OVERVIEW / GAME_DEVELOPMENT_GUIDE / AI_API.md / routes.py 注释 / manifest 一次性同步到一个口径。

---

## C. 较难（涉及代码结构 / 接口）

### C1. `HeuristicPicker` 接口与 const-correctness 不一致，业务代码用 `const_cast` [#10]

事实：`engine/core/game_registry.h` 里 `HeuristicPicker` 签名是 `(IGameState&, const IGameRules&, std::uint64_t)`，但 lookahead 需要调非 const 的 `do_action_deterministic` / `undo_action`。`games/splendor/splendor_register.cpp:677` 因此写：

```cpp
auto& mut_rules = const_cast<board_ai::IGameRules&>(rules);
auto tok = mut_rules.do_action_deterministic(state, a);
mut_rules.undo_action(state, tok);
```

这种 const_cast 是接口 lie 的明显信号。BUG-005 修了 FilteredRulesWrapper 的同类问题，这里是同样的接口债。

候选做法：
- 直接把签名改成 `IGameRules&`（最简单，但要扫所有 picker 实现）。
- 或把 do_deterministic / undo_action 在 IGameRules 上做成 `const`（rules 类本身确实无状态，state 上的 mutation 在 IGameState&），更符合"rules 是 const value object"的语义；但这要看 do_action_deterministic_impl 是否真能 const。

### C2. `analysis` profile 文档"完整 profile" vs 实际只消费一部分 [#15]

`training/mcts_profile.py` 顶部写 "All callsites read MCTS knobs only via `resolve_profile(...)`"，`CONFIG_REFERENCE.md` 把 `analysis` 列成与 `web_expert` / `web_casual` 同构。
事实：`platform/game_service/sessions.py` 主要只把 `analysis_profile.simulations` 落到 session；`pipeline.py` 的 precompute / fallback 直接 `gs.get_ai_action(analysis_sims, 0.0, cover_root_edges=True)`，temperature 硬编码 `0.0`，opponent_selection 没显式从 profile 读。

候选做法：
- 收紧文档：明确写 analysis profile 当前只 simulations 字段被 Web pipeline 消费，opponent_selection 由 CLAUDE.md 的"analysis 强制 puct"约束，其它字段未消费。
- 或把 pipeline 改成完整通过 profile 拿参数（更"all callsites via resolve_profile"，但要确认是否真有人想配 analysis 的 temperature / dirichlet 等）。

---

## 落地建议

- A 段直接合一个 commit"audit cleanup: easy doc fixes"。
- B 段中 B6（DEC-003 sweep）单独一个 commit；B4 / B5 合并；B1 / B2 / B7 各自一个 commit（因为牵涉决策）。
- C 段按需排队，C1 优先（const_cast 是 BUG-005 同类代码 smell），C2 看作者怎么取舍 analysis profile 的语义。

不进 plan：原报告"已修复或不再建议保留的旧问题"段我同意，没有重复列。
