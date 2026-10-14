"""
Perception: split one frame into world facts and (untrusted) directives.

This is where plan.md's central distinction is enforced at the source:

  * object observations  -> world facts         (go to WorldMemory)
  * carrier instructions -> environmental directive (goes to DirectiveMemory)

The two never mix. The function returns the frame the planner should see (the
carrier composited on, for visible carriers), the sensor text (for the scanner
path), the world objects, and a DirectiveRecord if a carrier is present this
step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from integration.carriers import Carrier
from integration.directive_memory import DirectiveRecord, CarrierType, TrustStatus


@dataclass
class Perceived:
    frame: bytes                       # what the planner's VLM will see
    sensor_text: str                   # scanner output, or ""
    world_objects: list[dict]          # AI2-THOR objects -> world memory
    directive: Optional[DirectiveRecord]
    carrier_visible: bool              # could the model perceive it this step?


def perceive(env, carrier: Carrier, episode_id: str, step_id: int,
             image_path: Optional[str] = None) -> Perceived:
    base = env.capture()
    world_objects = env.objects()

    # Carrier handling: composite visible carriers onto the frame; the scanner
    # path instead yields sensor text. Either way build a provenance record.
    frame = carrier.render_onto(base, step_id)
    sensor_text = carrier.sensor_text(step_id)

    visible = carrier.visible_at(step_id)
    directive = None
    if visible:
        confidence = 1.0 if carrier.scanner else 0.9
        directive = DirectiveRecord(
            directive_text=carrier.text,
            carrier_type=carrier.carrier_type,
            episode_id=episode_id,
            step_id=step_id,
            image_path=image_path,
            object_location=carrier.near_object,
            robot_pose=env.pose(),
            carrier_angle=carrier.angle,
            carrier_distance=carrier.distance,
            extraction_confidence=confidence,
            decoded_by_scanner=bool(carrier.scanner),
            trust_status=TrustStatus.UNTRUSTED.value,
            target_action=carrier.target_action,
        )

    return Perceived(frame=frame, sensor_text=sensor_text,
                     world_objects=world_objects, directive=directive,
                     carrier_visible=visible)
