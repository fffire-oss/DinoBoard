#pragma once

// Phase 1.6 (lightweight): observe-request envelope + post-event
// schema validation.
//
// `ObserveRequest` is the wire-level envelope for one observation
// (T2 in docs/plans/GOLDEN_STANDARD_IMPLEMENTATION.md appendix A).
// `actor` is REQUIRED (golden standard I17): the framework must know
// which seat caused the action so trackers can attribute observations
// correctly.
//
// `validate_post_events` is the single landing point for the
// schema's action_events table — golden standard §3.3 says the
// framework checks the post-event list against the schema before
// running the four-step apply_observation flow. If a mandatory event
// the schema declares for `action_id` is missing from `post_events`,
// the request is rejected with `MissingMandatoryEventError` rather
// than silently fed to the tracker (which would then compute on
// stale state and cause a hard-to-debug strength regression).
//
// Body of apply_observation itself lands per-game in Phase 3 once
// each game's schema + extract/apply virtuals are in place. Phase
// 1.6 ships only the type + the validator that doesn't depend on
// game-side reflection.

#include <stdexcept>
#include <string>
#include <vector>

#include "game_interfaces.h"     // PublicEvent, ActionId
#include "snapshot.h"
#include "visibility_schema.h"

namespace board_ai {

// One ObserveRequest from ground truth to an AI session. The session
// applies it via the four-step flow:
//   1. validate post_events against schema
//   2. state.viz_ ← snapshot.visibility_mask (full replace)
//   3. state typed payload ← snapshot.values (game-side)
//   4. tracker?.observe(actor, action_id, snapshot, post_events)
//
// Step 1 is enforced by validate_post_events below; steps 2-4 land
// per-game in Phase 3.
struct ObserveRequest {
  int actor = -1;                       // REQUIRED — golden standard I17.
                                        // -1 sentinel means "unset",
                                        // validate_request rejects it.
  ActionId action_id = 0;
  PerspectiveSnapshot snapshot;
  std::vector<PublicEvent> post_events;
};

// Thrown by validate_post_events when the schema's action_events
// table marks an event mandatory for `action_id` and the request
// did not include it. Caught at the binding boundary and surfaced
// as an HTTP 400 with the missing kind in the body.
class MissingMandatoryEventError : public std::runtime_error {
 public:
  MissingMandatoryEventError(ActionId action_id, const std::string& kind)
      : std::runtime_error(
            "post_events missing mandatory event '" + kind +
            "' for action_id " + std::to_string(action_id)),
        action_id_(action_id),
        kind_(kind) {}

  ActionId action_id() const { return action_id_; }
  const std::string& kind() const { return kind_; }

 private:
  ActionId action_id_;
  std::string kind_;
};

// Reject an ObserveRequest whose `actor` was not set (-1 sentinel).
// Cheap up-front check before any payload parsing; called by the
// binding before validate_post_events.
inline void validate_request_actor_set(const ObserveRequest& req) {
  if (req.actor < 0) {
    throw std::invalid_argument(
        "ObserveRequest: 'actor' is required (received " +
        std::to_string(req.actor) + ")");
  }
}

// Phase 1.6 stance on mandatory/optional: schema's EventDecl gains a
// boolean `mandatory` flag (default true — most events ARE required;
// optional is the exception). The pre-1.6 schema ships without the
// flag; for now we treat every declared event as mandatory. When a
// game starts using optional events (Phase 3 / 4), the EventDecl
// definition gets the flag and this validator gets the per-event
// branch.
//
// Validation is positional-tolerant: events may appear in any order
// in `post_events`. We just check every mandatory kind appears at
// least once. Extra event kinds (not declared by the schema) are
// allowed and ignored — the schema is a "must include these" floor,
// not a "these and only these" closure (so games can roll new event
// kinds in without breaking old recordings).
//
// Throws MissingMandatoryEventError on the FIRST missing mandatory
// event. Callers needing the full list of missing events can catch +
// re-call after the schema is amended; for the live request path,
// failing fast is fine — GT must amend its emitter, not paper over it.
inline void validate_post_events(
    const viz::VisibilitySchema& schema, ActionId action_id,
    const std::vector<PublicEvent>& post_events) {
  for (const auto& mapping : schema.action_events) {
    if (!mapping.matches) continue;
    if (!mapping.matches(static_cast<int>(action_id))) continue;
    // Found the action's declared event list. Check each declared
    // event appears in post_events at least once.
    for (const auto& decl : mapping.events) {
      // EventDecl::phase==1 → post-action (kPostAction). pre-action
      // events live in pre_events on a different request slot; this
      // validator skips them.
      if (decl.phase != 1) continue;
      bool found = false;
      for (const auto& evt : post_events) {
        if (evt.first == decl.kind) {
          found = true;
          break;
        }
      }
      if (!found) {
        throw MissingMandatoryEventError(action_id, decl.kind);
      }
    }
    // schema.action_events typically has one matching entry per
    // action_id range — first match wins, no need to keep iterating.
    return;
  }
  // No matching action_events entry → schema declares no required
  // post events for this action. Empty post_events is fine.
}

}  // namespace board_ai
