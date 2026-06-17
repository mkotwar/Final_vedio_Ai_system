from pydantic import BaseModel, Field

class SamplingMetrics(BaseModel):
    video_id: str
    video_duration_seconds: float = Field(default=0.0)
    total_frames_seen: int = Field(default=0)
    
    dropped_by_motion_window: int = Field(default=0)
    dropped_by_ssim: int = Field(default=0)
    dropped_by_histogram: int = Field(default=0)
    dropped_by_motion_threshold: int = Field(default=0)
    
    kept_frames: int = Field(default=0)
    motion_windows_detected: int = Field(default=0)
    
    reduction_percent: float = Field(default=0.0)
    vlm_calls_saved: int = Field(default=0)
    processing_duration_seconds: float = Field(default=0.0)
    
    # Dynamic FPS Telemetry
    mode_idle_seconds: float = Field(default=0.0)
    mode_low_seconds: float = Field(default=0.0)
    mode_normal_seconds: float = Field(default=0.0)
    mode_high_seconds: float = Field(default=0.0)
    mode_burst_seconds: float = Field(default=0.0)
    fps_transitions: int = Field(default=0)
    burst_activations: int = Field(default=0)
    average_extraction_fps: float = Field(default=0.0)
    
    # Event Candidate Layer Telemetry
    candidate_frames_generated: int = Field(default=0)
    candidate_frames_rejected: int = Field(default=0)
    candidate_frames_sent_to_vlm: int = Field(default=0)
    candidate_reduction_percent: float = Field(default=0.0)
    average_candidate_density: float = Field(default=0.0)
    sampling_collapse: bool = Field(default=False)
