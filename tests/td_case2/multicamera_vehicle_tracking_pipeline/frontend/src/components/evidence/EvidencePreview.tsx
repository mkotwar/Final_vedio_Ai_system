import { useEffect, useMemo, useState } from 'react'
import { getMediaSignedUrl } from '../../api/media'
import { resolveApiAssetUrl } from '../../api/client'
import type { MediaReference } from '../../types/media'

interface EvidencePreviewProps {
  media?: MediaReference | null
  title?: string
  buttonLabel?: string
  viewportClassName?: string
  imageClassName?: string
  placeholderClassName?: string
  placeholderTitle?: string
  placeholderText?: string
}

export default function EvidencePreview({
  media,
  title,
  buttonLabel,
  viewportClassName,
  imageClassName,
  placeholderClassName,
  placeholderTitle = 'Image unavailable',
  placeholderText,
}: EvidencePreviewProps) {
  const [signedUrl, setSignedUrl] = useState<string | null>(null)
  const [isResolving, setIsResolving] = useState(false)
  const [loadFailed, setLoadFailed] = useState(false)
  const [isPreviewOpen, setIsPreviewOpen] = useState(false)

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
        if (isMounted) {
          setSignedUrl(response.url || null)
        }
      })
      .catch(() => {
        if (isMounted) {
          setSignedUrl(null)
        }
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

  useEffect(() => {
    if (!isPreviewOpen) {
      return
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        setIsPreviewOpen(false)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isPreviewOpen])

  const imageUrl = useMemo(() => {
    if (media?.availability === 'LOCAL_FILE' && media.content_url) {
      return resolveApiAssetUrl(media.content_url)
    }
    if (media?.availability === 'SIGNED_URL' && signedUrl) {
      return signedUrl
    }
    return null
  }, [media?.availability, media?.content_url, signedUrl])

  const resolvedPlaceholder = placeholderText || getPlaceholderText(media, loadFailed, isResolving)
  const showImage = Boolean(imageUrl) && !loadFailed
  const previewTitle = title || media?.media_type || 'Evidence'
  const previewButtonLabel = buttonLabel || `Open ${previewTitle} preview`

  return (
    <>
      {showImage ? (
        <button
          type="button"
          className={viewportClassName || 'evidence-card__image-button'}
          onClick={() => setIsPreviewOpen(true)}
          aria-label={previewButtonLabel}
        >
          <img
            src={imageUrl || undefined}
            alt={previewTitle}
            className={imageClassName || 'evidence-card__image'}
            onError={() => setLoadFailed(true)}
          />
        </button>
      ) : (
        <div
          className={placeholderClassName || viewportClassName || 'evidence-card__placeholder'}
          role="img"
          aria-label={resolvedPlaceholder}
        >
          <strong>{placeholderTitle}</strong>
          <p>{resolvedPlaceholder}</p>
        </div>
      )}

      {showImage && isPreviewOpen ? (
        <div className="evidence-modal" role="dialog" aria-modal="true" aria-label={`${previewTitle} preview`}>
          <button
            type="button"
            className="evidence-modal__backdrop"
            aria-label="Close preview"
            onClick={() => setIsPreviewOpen(false)}
          />
          <div className="evidence-modal__panel">
            <div className="evidence-modal__header">
              <div>
                <h4>{previewTitle}</h4>
                <p className="table-subtext">{media?.availability || 'REFERENCE_ONLY'}</p>
              </div>
              <button type="button" className="button button--secondary" onClick={() => setIsPreviewOpen(false)}>
                Close
              </button>
            </div>
            <div className="evidence-modal__viewport">
              <img src={imageUrl || undefined} alt={previewTitle} className="evidence-modal__image" />
            </div>
          </div>
        </div>
      ) : null}
    </>
  )
}

export function getPlaceholderText(
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
