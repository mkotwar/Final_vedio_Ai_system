interface CameraStep {
  camera_code?: string | null
  track_uuid?: string | null
}

interface CameraSequenceProps {
  steps: CameraStep[]
}

export default function CameraSequence({ steps }: CameraSequenceProps) {
  return (
    <div className="sequence" aria-label="Camera sequence">
      {steps.map((step, index) => (
        <div key={`${step.track_uuid || 'step'}-${index}`} className="sequence__step">
          <div className="sequence__card">
            <p className="sequence__camera">{step.camera_code || 'Unknown camera'}</p>
            <p className="sequence__track">{step.track_uuid || 'Unknown track'}</p>
          </div>
          {index < steps.length - 1 ? <div className="sequence__arrow">↓</div> : null}
        </div>
      ))}
    </div>
  )
}
