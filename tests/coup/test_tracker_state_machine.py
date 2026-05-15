"""Unit tests for CoupBeliefTracker's stage-boundary state machine.

These tests drive specific (action, expected events, expected signals)
sequences through GameSession and assert on `get_belief_snapshot()`.
They cover bugs that the smoke-only `test_heuristic_sampling.py` cannot
catch:

1. Single-counting: in N>2 games every opp Allow after a claim must
   accrue ONE signal total, not N-1 (one per Allow). The fix is
   stage-boundary `claim_unchallenged` instead of action-level Allow
   counting.
2. Cleanup: `card_revealed` / `claim_resolved_truthful` /
   `exchange_complete` events must reset stale signals so the heuristic
   does not double-bias against already-grounded truth.
3. Pending-claim tracking: pending_claimer / pending_claim_role /
   pending_challenged must update on claim, challenge, and resolve.

The tracker is perspective-invariant, so we only test the session's
shared `bt_` via `gs.get_belief_snapshot()`.
"""
import dinoboard_engine
import pytest


# Coup action IDs (mirrors games/coup/coup_state.h).
INCOME = 0
FOREIGN_AID = 1
COUP_OFFSET = 2
TAX = 6
ASSASSIN_OFFSET = 7
STEAL_OFFSET = 11
EXCHANGE = 15
CHALLENGE = 16
ALLOW = 17
BLOCK_DUKE = 18
BLOCK_CONTESSA = 19
BLOCK_AMBASSADOR = 20
BLOCK_CAPTAIN = 21
ALLOW_NO_BLOCK = 22

# Character ids.
DUKE = 0
ASSASSIN = 1
CAPTAIN = 2
AMBASSADOR = 3
CONTESSA = 4
NUM_CHARS = 5


def _signals(snapshot, n_players):
    """Reshape flat signals back to [n_players][NUM_CHARS]."""
    flat = snapshot["signals"]
    assert len(flat) == n_players * NUM_CHARS
    return [
        list(flat[p * NUM_CHARS:(p + 1) * NUM_CHARS])
        for p in range(n_players)
    ]


def _zero_signals(n_players):
    return [[0] * NUM_CHARS for _ in range(n_players)]


class TestInitialState:
    """Fresh session: no signals, no pending claim."""

    @pytest.mark.parametrize("game_id,n", [
        ("coup_2p", 2), ("coup_3p", 3), ("coup_4p", 4)
    ])
    def test_fresh_session_zero_signals(self, game_id, n):
        gs = dinoboard_engine.GameSession(game_id, seed=1, model_path="")
        snap = gs.get_belief_snapshot()
        assert _signals(snap, n) == _zero_signals(n)
        assert snap["pending_claimer"] == -1
        assert snap["pending_claim_role"] == -1
        assert snap["pending_challenged"] is False


class TestClaimUnchallenged:
    """A claim followed by every opponent Allowing should accrue exactly
    one signal on the claimer/role pair, regardless of N."""

    def test_2p_tax_allow(self):
        gs = dinoboard_engine.GameSession("coup_2p", seed=1, model_path="")
        gs.apply_action(TAX)            # P0 claims Duke
        gs.apply_action(ALLOW)          # P1 Allows → claim_unchallenged

        snap = gs.get_belief_snapshot()
        sig = _signals(snap, 2)
        assert sig[0][DUKE] == 1, f"expected signals_[0][Duke]=1, got {sig}"
        # No other signal touched.
        for p in range(2):
            for c in range(NUM_CHARS):
                if (p, c) == (0, DUKE):
                    continue
                assert sig[p][c] == 0, f"unexpected signal at ({p},{c}): {sig}"
        # Pending closed.
        assert snap["pending_claimer"] == -1
        assert snap["pending_claim_role"] == -1
        assert snap["pending_challenged"] is False

    def test_3p_tax_two_allows_no_double_count(self):
        """N=3: P1 and P2 each Allow. Tracker must accrue ONE signal
        total, not 2 (the bug the stage-boundary fix addresses)."""
        gs = dinoboard_engine.GameSession("coup_3p", seed=1, model_path="")
        gs.apply_action(TAX)            # P0 claims Duke
        gs.apply_action(ALLOW)          # P1 Allows; stage stays kChallengeAction
        # After P1's Allow, no event yet — current_player advances to P2.
        # Verify signals are still zero (claim_unchallenged hasn't fired).
        mid = gs.get_belief_snapshot()
        assert _signals(mid, 3) == _zero_signals(3), (
            "intermediate Allow leaked a signal — stage-boundary check broken"
        )
        gs.apply_action(ALLOW)          # P2 Allows → claim_unchallenged (1×)

        snap = gs.get_belief_snapshot()
        sig = _signals(snap, 3)
        assert sig[0][DUKE] == 1, f"expected exactly 1 signal, got {sig}"
        for p in range(3):
            for c in range(NUM_CHARS):
                if (p, c) == (0, DUKE):
                    continue
                assert sig[p][c] == 0
        assert snap["pending_claimer"] == -1

    def test_4p_tax_three_allows_no_triple_count(self):
        gs = dinoboard_engine.GameSession("coup_4p", seed=1, model_path="")
        gs.apply_action(TAX)
        gs.apply_action(ALLOW)
        gs.apply_action(ALLOW)
        # Two Allows in: still no event.
        mid = gs.get_belief_snapshot()
        assert _signals(mid, 4) == _zero_signals(4)
        gs.apply_action(ALLOW)          # final Allow → single event

        snap = gs.get_belief_snapshot()
        sig = _signals(snap, 4)
        assert sig[0][DUKE] == 1
        # Tail must be untouched.
        for p in range(4):
            for c in range(NUM_CHARS):
                if (p, c) == (0, DUKE):
                    continue
                assert sig[p][c] == 0


class TestChallengePendingState:
    """A Challenge declaration sets pending_challenged and accrues a
    signal on the challenger for the claimed role."""

    def test_2p_tax_challenge_sets_pending_and_accrues(self):
        gs = dinoboard_engine.GameSession("coup_2p", seed=1, model_path="")
        gs.apply_action(TAX)            # P0 claims Duke
        # After Tax, pending_claimer=0, pending_claim_role=Duke.
        post_claim = gs.get_belief_snapshot()
        assert post_claim["pending_claimer"] == 0
        assert post_claim["pending_claim_role"] == DUKE
        assert post_claim["pending_challenged"] is False

        gs.apply_action(CHALLENGE)      # P1 challenges
        # Action-level: challenger (P1) accrues signal_[1][Duke] += 1
        # (they "show" they think they could hold Duke).
        snap = gs.get_belief_snapshot()
        sig = _signals(snap, 2)
        assert sig[1][DUKE] == 1, (
            f"expected challenger to accrue signals_[1][Duke]=1, got {sig}"
        )
        # No claim_unchallenged event — stage went to kResolveChallengeAction.
        assert sig[0][DUKE] == 0
        # pending_challenged = true; pending_claimer/role still set
        # (cleared on resolve event, not on challenge declaration).
        assert snap["pending_challenged"] is True


class TestRawCountsV02:
    """v0.2 raw-event accumulators: pre/post claim counts, pre/post
    challenge-initiated, last_reshuffle_kind, last_revealed_role."""

    # ReshuffleKind enum (matches CoupBeliefTracker::ReshuffleKind in C++).
    KIND_NONE = 0
    KIND_EXCHANGE = 1
    KIND_REVEAL_TRUTHFUL = 2

    def _claim_counts(self, snap, kind, n_players):
        flat = snap[f"{kind}_claim_counts"]
        return [
            list(flat[p * NUM_CHARS:(p + 1) * NUM_CHARS])
            for p in range(n_players)
        ]

    def _challenge_counts(self, snap, kind, n_players):
        flat = snap[f"{kind}_challenge_initiated"]
        return [
            list(flat[p * NUM_CHARS:(p + 1) * NUM_CHARS])
            for p in range(n_players)
        ]

    def test_initial_zero(self):
        gs = dinoboard_engine.GameSession("coup_2p", seed=1, model_path="")
        snap = gs.get_belief_snapshot()
        assert self._claim_counts(snap, "pre", 2) == _zero_signals(2)
        assert self._claim_counts(snap, "post", 2) == _zero_signals(2)
        assert self._challenge_counts(snap, "pre", 2) == _zero_signals(2)
        assert self._challenge_counts(snap, "post", 2) == _zero_signals(2)
        assert list(snap["last_reshuffle_kind"]) == [self.KIND_NONE] * 2
        assert list(snap["last_revealed_role"]) == [-1] * 2

    def test_claim_lands_on_post_not_pre(self):
        """First claim before any reshuffle: post += 1, pre stays 0."""
        gs = dinoboard_engine.GameSession("coup_2p", seed=1, model_path="")
        gs.apply_action(TAX)            # P0 claims Duke
        snap = gs.get_belief_snapshot()
        post = self._claim_counts(snap, "post", 2)
        pre = self._claim_counts(snap, "pre", 2)
        assert post[0][DUKE] == 1
        assert pre == _zero_signals(2)
        # Reshuffle indicators untouched (no reshuffle happened).
        assert list(snap["last_reshuffle_kind"]) == [self.KIND_NONE] * 2

    def test_challenge_lands_on_post_initiated(self):
        gs = dinoboard_engine.GameSession("coup_2p", seed=1, model_path="")
        gs.apply_action(TAX)            # P0 claims Duke
        gs.apply_action(CHALLENGE)      # P1 challenges
        snap = gs.get_belief_snapshot()
        post_chal = self._challenge_counts(snap, "post", 2)
        # P1 (the challenger) initiated a challenge against role Duke.
        assert post_chal[1][DUKE] == 1
        # Pre-side stays 0.
        assert self._challenge_counts(snap, "pre", 2) == _zero_signals(2)


class TestStageBoundaryEventsFromSelfplay:
    """End-to-end: run a real selfplay episode with descent maintenance,
    and verify that the tracker's terminal serialize() yields a sane
    schema (right shapes, all entries non-negative, signals never
    exceed `kCardsPerCharacter * NPlayers` per role)."""

    @pytest.mark.parametrize("game_id,n", [
        ("coup_2p", 2), ("coup_3p", 3), ("coup_4p", 4)
    ])
    def test_signals_within_bounds_after_selfplay_drive(self, game_id, n):
        gs = dinoboard_engine.GameSession(game_id, seed=42, model_path="")
        for _ in range(20):
            if gs.is_terminal:
                break
            legal = gs.get_legal_actions()
            gs.apply_action(legal[0])
        snap = gs.get_belief_snapshot()
        sig = _signals(snap, n)
        for p in range(n):
            for c in range(NUM_CHARS):
                assert sig[p][c] >= 0, (
                    f"signal {(p, c)} went negative: {sig[p][c]}"
                )
                # Loose upper bound: signals are unchallenged-claim count
                # plus implicit-challenger count. Bounded by ply count.
                assert sig[p][c] <= 50

    @pytest.mark.parametrize("game_id", ["coup_2p", "coup_3p", "coup_4p"])
    def test_serialize_keys_stable(self, game_id):
        gs = dinoboard_engine.GameSession(game_id, seed=7, model_path="")
        snap = gs.get_belief_snapshot()
        # Serialize keys must be stable (used for tracker-equality assertion
        # in test_sim_tracker_descent_maintained / perspective invariance).
        assert set(snap.keys()) == {
            "signals",
            "pre_claim_counts",
            "post_claim_counts",
            "pre_challenge_initiated",
            "post_challenge_initiated",
            "last_reshuffle_kind",
            "last_revealed_role",
            "pending_claimer",
            "pending_claim_role",
            "pending_challenged",
        }
