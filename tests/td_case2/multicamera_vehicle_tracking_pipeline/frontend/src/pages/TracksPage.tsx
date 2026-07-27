import { useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { listTracks } from '../api/tracks'
import type { TrackListItem } from '../types/track'
import Pagination from '../components/tables/Pagination'
import FilterBar from '../components/filters/FilterBar'
import LoadingState from '../components/states/LoadingState'
import ErrorState from '../components/states/ErrorState'
import EmptyState from '../components/states/EmptyState'
import TrackSummaryCard from '../components/cards/TrackSummaryCard'

export default function TracksPage() {
  const [searchParams, setSearchParams] = useSearchParams()

  const filters = useMemo(
    () => ({
      runCode: searchParams.get('run_code') || '',
      page: Number(searchParams.get('page') || '1'),
      page_size: Number(searchParams.get('page_size') || '10'),
      camera_code: searchParams.get('camera_code') || undefined,
      vehicle_class: searchParams.get('vehicle_class') || undefined,
      colour: searchParams.get('colour') || undefined,
      plate: searchParams.get('plate') || undefined,
      plate_status: searchParams.get('plate_status') || undefined,
      lifecycle_state: searchParams.get('lifecycle_state') || undefined,
      minimum_confidence: searchParams.get('minimum_confidence') || '0.5',
      has_media: searchParams.get('has_media') || undefined,
      sort_by: searchParams.get('sort_by') || 'first_seen_at',
      sort_order: searchParams.get('sort_order') || 'desc',
    }),
    [searchParams],
  )

  const tracksQuery = useQuery({
    queryKey: ['tracks', filters],
    queryFn: () =>
      listTracks(filters.runCode, {
        page: filters.page,
        page_size: filters.page_size,
        camera_code: filters.camera_code,
        vehicle_class: filters.vehicle_class,
        colour: filters.colour,
        plate: filters.plate,
        plate_status: filters.plate_status,
        lifecycle_state: filters.lifecycle_state,
        minimum_confidence: filters.minimum_confidence,
        has_media: filters.has_media,
        sort_by: filters.sort_by,
        sort_order: filters.sort_order,
      }),
    enabled: Boolean(filters.runCode),
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

  return (
    <div className="page-stack">
      <FilterBar>
        <label>
          <span>Run code</span>
          <input value={filters.runCode} onChange={(event) => updateFilter('run_code', event.target.value)} placeholder="RUN_20260724_151402" />
        </label>
        <label>
          <span>Camera code</span>
          <input value={filters.camera_code || ''} onChange={(event) => updateFilter('camera_code', event.target.value)} placeholder="CAM_001" />
        </label>
        <label>
          <span>Vehicle class</span>
          <input value={filters.vehicle_class || ''} onChange={(event) => updateFilter('vehicle_class', event.target.value)} placeholder="CAR" />
        </label>
        <label>
          <span>Colour</span>
          <input value={filters.colour || ''} onChange={(event) => updateFilter('colour', event.target.value)} placeholder="GREY" />
        </label>
        <label>
          <span>Plate</span>
          <input value={filters.plate || ''} onChange={(event) => updateFilter('plate', event.target.value)} placeholder="DL8CBF6268" />
        </label>
        <label>
          <span>Plate status</span>
          <input value={filters.plate_status || ''} onChange={(event) => updateFilter('plate_status', event.target.value)} placeholder="VERIFIED" />
        </label>
        <label>
          <span>Lifecycle</span>
          <input value={filters.lifecycle_state || ''} onChange={(event) => updateFilter('lifecycle_state', event.target.value)} placeholder="COMPLETED" />
        </label>
        <label>
          <span>Min confidence</span>
          <input value={filters.minimum_confidence || ''} onChange={(event) => updateFilter('minimum_confidence', event.target.value)} placeholder="0.50" type="number" min="0" max="1" step="0.01" />
        </label>
        <label>
          <span>Has media</span>
          <select value={filters.has_media || ''} onChange={(event) => updateFilter('has_media', event.target.value)}>
            <option value="">Any</option>
            <option value="true">Yes</option>
            <option value="false">No</option>
          </select>
        </label>
        <button className="button button--secondary" onClick={() => void tracksQuery.refetch()} type="button">
          Refresh
        </button>
      </FilterBar>

      {!filters.runCode ? (
        <EmptyState title="Run code required" description="Enter a processing run code to load local tracks." />
      ) : tracksQuery.isPending ? (
        <LoadingState label="Loading local tracks..." />
      ) : tracksQuery.isError ? (
        <ErrorState title="Tracks unavailable" message="The selected run's local tracks could not be loaded." onRetry={() => void tracksQuery.refetch()} />
      ) : tracksQuery.data.items.length === 0 ? (
        <EmptyState title="No tracks found" description="Try broadening the filters for the selected run." />
      ) : (
        <>
          <div className="stack-list">
            {tracksQuery.data.items.map((row: TrackListItem) => (
              <TrackSummaryCard key={row.track_uuid} track={row} detailHref={`/tracks/${encodeURIComponent(row.track_uuid)}`} />
            ))}
          </div>
          <Pagination
            page={tracksQuery.data.page}
            pageSize={tracksQuery.data.page_size}
            total={tracksQuery.data.total}
            hasNext={tracksQuery.data.has_next}
            onPageChange={(page) => updateFilter('page', String(page))}
          />
        </>
      )}
    </div>
  )
}
