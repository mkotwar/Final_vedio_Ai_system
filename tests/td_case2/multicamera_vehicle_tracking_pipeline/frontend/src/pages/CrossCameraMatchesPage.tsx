import type { ReactNode } from 'react'
import { useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { listMatches } from '../api/matches'
import type { MatchListItem } from '../types/match'
import FilterBar from '../components/filters/FilterBar'
import Pagination from '../components/tables/Pagination'
import LoadingState from '../components/states/LoadingState'
import ErrorState from '../components/states/ErrorState'
import EmptyState from '../components/states/EmptyState'
import StatusBadge from '../components/states/StatusBadge'
import ConfidenceBadge from '../components/states/ConfidenceBadge'
import TrackSummaryCard from '../components/cards/TrackSummaryCard'
import type { TrackListItem } from '../types/track'

export default function CrossCameraMatchesPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const filters = useMemo(
    () => ({
      run_code: searchParams.get('run_code') || undefined,
      decision: searchParams.get('decision') || undefined,
      minimum_score: searchParams.get('minimum_score') || undefined,
      rule_version: searchParams.get('rule_version') || undefined,
      camera_code: searchParams.get('camera_code') || undefined,
      page: Number(searchParams.get('page') || '1'),
      page_size: Number(searchParams.get('page_size') || '10'),
      sort_by: searchParams.get('sort_by') || 'updated_at',
      sort_order: searchParams.get('sort_order') || 'desc',
    }),
    [searchParams],
  )

  const matchesQuery = useQuery({
    queryKey: ['matches', filters],
    queryFn: () => listMatches(filters),
  })

  function updateFilter(name: string, value: string) {
    const next = new URLSearchParams(searchParams)
    if (value) {
      next.set(name, value)
    } else {
      next.delete(name)
    }
    if (name !== 'page') {
      next.set('page', '1')
    }
    setSearchParams(next)
  }

  if (matchesQuery.isPending) {
    return <LoadingState label="Loading cross-camera matches..." />
  }

  if (matchesQuery.isError) {
    return <ErrorState title="Matches unavailable" message="The frontend could not load cross-camera match results." onRetry={() => void matchesQuery.refetch()} />
  }

  return (
    <div className="page-stack">
      <FilterBar>
        <label>
          <span>Run code</span>
          <input value={filters.run_code || ''} onChange={(event) => updateFilter('run_code', event.target.value)} placeholder="RUN_20260724_151402" />
        </label>
        <label>
          <span>Decision</span>
          <input value={filters.decision || ''} onChange={(event) => updateFilter('decision', event.target.value)} placeholder="CONFIRMED" />
        </label>
        <label>
          <span>Minimum score</span>
          <input value={filters.minimum_score || ''} onChange={(event) => updateFilter('minimum_score', event.target.value)} placeholder="0.9" type="number" min="0" max="1" step="0.01" />
        </label>
        <label>
          <span>Rule version</span>
          <input value={filters.rule_version || ''} onChange={(event) => updateFilter('rule_version', event.target.value)} placeholder="global_match_v1" />
        </label>
        <label>
          <span>Camera code</span>
          <input value={filters.camera_code || ''} onChange={(event) => updateFilter('camera_code', event.target.value)} placeholder="CAM_001" />
        </label>
        <button className="button button--secondary" onClick={() => void matchesQuery.refetch()} type="button">
          Refresh
        </button>
      </FilterBar>

      {matchesQuery.data.items.length === 0 ? (
        <EmptyState title="No matches found" description="No cross-camera matches matched the current filters." />
      ) : (
        <>
          <div className="stack-list">
            {matchesQuery.data.items.map((match) => (
              <article className="match-comparison-card" key={match.id || `${match.source_track_uuid}-${match.candidate_track_uuid}`}>
                <div className="match-comparison-card__header">
                  <div>
                    <p className="summary-card__eyebrow">Cross-camera match</p>
                    <h3>{match.linked_global_vehicle_code || `${match.source_track_uuid || 'Source'} → ${match.candidate_track_uuid || 'Candidate'}`}</h3>
                  </div>
                  <div className="match-comparison-card__scores">
                    <StatusBadge value={match.decision} />
                    <ConfidenceBadge value={match.overall_score} />
                  </div>
                </div>

                <div className="match-comparison-card__grid">
                  <TrackSummaryCard
                    track={toTrackSummary(match.source_track, match.source_track_uuid, match.source_camera_code)}
                    detailHref={match.source_track_uuid ? `/tracks/${encodeURIComponent(match.source_track_uuid)}` : undefined}
                  />
                  <TrackSummaryCard
                    track={toTrackSummary(match.candidate_track, match.candidate_track_uuid, match.candidate_camera_code)}
                    detailHref={match.candidate_track_uuid ? `/tracks/${encodeURIComponent(match.candidate_track_uuid)}` : undefined}
                  />
                </div>

                <dl className="match-comparison-card__details">
                  <Meta label="Plate score" value={<ConfidenceBadge value={match.plate_score} />} />
                  <Meta label="Class score" value={<ConfidenceBadge value={match.class_score} />} />
                  <Meta label="Colour score" value={<ConfidenceBadge value={match.colour_score} />} />
                  <Meta label="Route score" value={<ConfidenceBadge value={match.route_score} />} />
                  <Meta label="Time score" value={<ConfidenceBadge value={match.time_score} />} />
                  <Meta label="Visual score" value={<ConfidenceBadge value={match.visual_score} />} />
                  <Meta label="Rule version" value={match.rule_version || 'N/A'} />
                  <Meta label="Reasons" value={match.decision_reasons.join(', ') || 'N/A'} wide />
                </dl>
              </article>
            ))}
          </div>
          <Pagination
            page={matchesQuery.data.page}
            pageSize={matchesQuery.data.page_size}
            total={matchesQuery.data.total}
            hasNext={matchesQuery.data.has_next}
            onPageChange={(page) => updateFilter('page', String(page))}
          />
        </>
      )}
    </div>
  )
}

function toTrackSummary(track: MatchListItem['source_track'] | MatchListItem['candidate_track'], trackUuid?: string | null, cameraCode?: string | null): TrackListItem {
  return {
    track_uuid: track?.track_uuid || trackUuid || 'Unknown track',
    camera_code: track?.camera_code || cameraCode || null,
    vehicle_class: track?.vehicle_class || null,
    lifecycle_state: track?.lifecycle_state || null,
    first_seen_at: track?.first_seen_at || null,
    last_seen_at: track?.last_seen_at || null,
    best_detection_confidence: track?.best_detection_confidence || null,
    primary_colour: track?.primary_colour || null,
    plate_result: track?.plate_result || null,
    canonical_plate: track?.canonical_plate || null,
    plate_status: track?.plate_status || null,
    plate_confidence: track?.plate_confidence || null,
    primary_media: track?.primary_media || null,
    primary_vehicle_media: track?.primary_vehicle_media || null,
    primary_plate_media: track?.primary_plate_media || null,
  }
}

function Meta({ label, value, wide = false }: { label: string; value: ReactNode; wide?: boolean }) {
  return (
    <div className={wide ? 'match-comparison-card__detail match-comparison-card__detail--wide' : 'match-comparison-card__detail'}>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  )
}
