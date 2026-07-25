import { useEffect, useMemo, useState } from 'react'
import { getMediaSignedUrl } from '../../api/media'
import { resolveApiAssetUrl } from '../../api/client'
import type { MediaReference } from '../../types/media'

interface EvidenceCardProps {
  media?: MediaReference | null
}

export default function EvidenceCard({ media }: EvidenceCardProps) {
  const [signedUrl, setSignedUrl] = useState<string | null>(null)
  const [isResolving, setIsResolving] = useState(false)
  const [loadFailed, setLoadFailed] = useState(false)

  useEffect(() => {
    let isMounted = true
    setSignedUrl(null)
    setLoadFailed(false)

    if (!media?.media_id || media.availability !== 'SIGNED_URL') {
      setIsResolving(false)
      return
    }

    setIsResolving(true)
    void getMediaSignedUrl(media.media_id)
      .then((response) => {
        if (!isMounted) {
          return
        }
        setSignedUrl(response.url || null)
      })
      .catch(() => {
        if (!isMounted) {
          return
        }
        setSignedUrl(null)
      })
      .finally(() => {
        if (isMounted) {
          setIsResolving(false)
        }
      })

    return () => {
      isMounted = false
    }
  }, [media?.availability, media?.media_id])

  const imageUrl = useMemo(() => {
    if (media?.availability === 'LOCAL_FILE' && media.content_url) {
      return resolveApiAssetUrl(media.content_url)
    }
    if (media?.availability === 'SIGNED_URL' && signedUrl) {
      return signedUrl
    }
    return null
  }, [media?.availability, media?.content_url, signedUrl])

  const placeholderText = getPlaceholderText(media, loadFailed, isResolving)
  const showImage = Boolean(imageUrl) && !loadFailed

  return (
    <article className="evidence-card">
      <div className="evidence-card__header">
        <div>
          <h4>{media?.media_type || 'Evidence'}</h4>
          <p className="table-subtext">{media?.availability || 'REFERENCE_ONLY'}</p>
        </div>
        <div className="evidence-card__badges">
          {media?.is_primary ? <span className="badge badge--success">PRIMARY</span> : null}
          <span className="badge badge--neutral">{media?.availability || 'REFERENCE_ONLY'}</span>
        </div>
      </div>

      <div className="evidence-card__preview">
        {showImage ? (
          <a href={imageUrl || '#'} target="_blank" rel="noreferrer" className="evidence-card__image-link">
            <img
              src={imageUrl || undefined}
              alt={media?.media_type || 'Evidence image'}
              className="evidence-card__image"
              onError={() => setLoadFailed(true)}
            />
          </a>
        ) : (
          <div className="evidence-card__placeholder" role="img" aria-label={placeholderText}>
            <strong>Image unavailable</strong>
            <p>{placeholderText}</p>
          </div>
        )}
      </div>

      <dl className="meta-grid">
        <div>
          <dt>Frame</dt>
          <dd>{media?.frame_number ?? 'N/A'}</dd>
        </div>
        <div>
          <dt>Quality</dt>
          <dd>{typeof media?.quality_score === 'number' ? media.quality_score.toFixed(2) : 'N/A'}</dd>
        </div>
        <div>
          <dt>Selection rank</dt>
          <dd>{media?.selection_rank ?? 'N/A'}</dd>
        </div>
        <div>
          <dt>Size</dt>
          <dd>{media?.width && media?.height ? `${media.width} × ${media.height}` : 'N/A'}</dd>
        </div>
      </dl>
    </article>
  )
}

function getPlaceholderText(
  media: MediaReference | null | undefined,
  loadFailed: boolean,
  isResolving: boolean,
) {
  if (loadFailed) {
    return 'The evidence image could not be loaded.'
  }
  if (isResolving) {
    return 'Loading image...'
  }
  switch (media?.availability) {
    case 'REFERENCE_ONLY':
      return 'Reference only'
    case 'MISSING':
      return 'Local evidence file not found'
    case 'UNSAFE_REFERENCE':
      return 'Unsafe media reference blocked'
    case 'UNSUPPORTED_PROVIDER':
      return 'Unsupported media provider'
    case 'SIGNED_URL':
      return 'Signed URL unavailable'
    default:
      return 'No media reference available.'
  }
}
