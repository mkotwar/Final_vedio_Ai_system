import { useMemo } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { listGlobalVehicles } from '../api/globalVehicles'
import type { GlobalVehicleListItem } from '../types/globalVehicle'
import FilterBar from '../components/filters/FilterBar'
import DataTable from '../components/tables/DataTable'
import Pagination from '../components/tables/Pagination'
import LoadingState from '../components/states/LoadingState'
import ErrorState from '../components/states/ErrorState'
import EmptyState from '../components/states/EmptyState'
import StatusBadge from '../components/states/StatusBadge'
import ConfidenceBadge from '../components/states/ConfidenceBadge'

export default function GlobalVehiclesPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const navigate = useNavigate()

  const filters = useMemo(
    () => ({
      run_code: searchParams.get('run_code') || undefined,
      status: searchParams.get('status') || undefined,
      vehicle_class: searchParams.get('vehicle_class') || undefined,
      colour: searchParams.get('colour') || undefined,
      plate: searchParams.get('plate') || undefined,
      minimum_confidence: searchParams.get('minimum_confidence') || undefined,
      minimum_camera_count: searchParams.get('minimum_camera_count') || undefined,
      page: Number(searchParams.get('page') || '1'),
      page_size: Number(searchParams.get('page_size') || '10'),
      sort_by: searchParams.get('sort_by') || 'first_seen_at',
      sort_order: searchParams.get('sort_order') || 'desc',
    }),
    [searchParams],
  )

  const globalVehiclesQuery = useQuery({
    queryKey: ['global-vehicles', filters],
    queryFn: () => listGlobalVehicles(filters),
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

  if (globalVehiclesQuery.isPending) {
    return <LoadingState label="Loading global vehicles..." />
  }

  if (globalVehiclesQuery.isError) {
    return (
      <ErrorState
        title="Global vehicles unavailable"
        message="The frontend could not load global vehicle objects."
        onRetry={() => void globalVehiclesQuery.refetch()}
      />
    )
  }

  return (
    <div className="page-stack">
      <FilterBar>
        <label>
          <span>Run code</span>
          <input value={filters.run_code || ''} onChange={(event) => updateFilter('run_code', event.target.value)} placeholder="RUN_20260724_151402" />
        </label>
        <label>
          <span>Status</span>
          <input value={filters.status || ''} onChange={(event) => updateFilter('status', event.target.value)} placeholder="CONFIRMED" />
        </label>
        <label>
          <span>Class</span>
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
          <span>Min confidence</span>
          <input value={filters.minimum_confidence || ''} onChange={(event) => updateFilter('minimum_confidence', event.target.value)} placeholder="0.95" type="number" min="0" max="1" step="0.01" />
        </label>
        <label>
          <span>Min cameras</span>
          <input value={filters.minimum_camera_count || ''} onChange={(event) => updateFilter('minimum_camera_count', event.target.value)} placeholder="2" type="number" min="1" />
        </label>
        <button className="button button--secondary" onClick={() => void globalVehiclesQuery.refetch()} type="button">
          Refresh
        </button>
      </FilterBar>

      {globalVehiclesQuery.data.items.length === 0 ? (
        <EmptyState title="No global vehicles found" description="Try widening the filters or selecting a run with persisted global objects." />
      ) : (
        <>
          <DataTable
            columns={[
              {
                key: 'code',
                header: 'Global vehicle code',
                render: (row: GlobalVehicleListItem) => (
                  <div>
                    <strong>{row.global_vehicle_code}</strong>
                    <p className="table-subtext">
                      {row.camera_count && row.camera_count > 1 ? 'Multi-camera object' : 'Single-camera object'}
                    </p>
                  </div>
                ),
              },
              { key: 'status', header: 'Status', render: (row: GlobalVehicleListItem) => <StatusBadge value={row.status} /> },
              { key: 'plate', header: 'Canonical plate', render: (row: GlobalVehicleListItem) => row.canonical_plate || 'N/A' },
              { key: 'colour', header: 'Canonical colour', render: (row: GlobalVehicleListItem) => row.canonical_colour || 'N/A' },
              { key: 'class', header: 'Canonical class', render: (row: GlobalVehicleListItem) => row.canonical_vehicle_class || 'N/A' },
              { key: 'confidence', header: 'Confidence', render: (row: GlobalVehicleListItem) => <ConfidenceBadge value={row.confidence} /> },
              { key: 'cameras', header: 'Cameras', render: (row: GlobalVehicleListItem) => row.camera_count ?? 'N/A' },
              { key: 'tracks', header: 'Tracks', render: (row: GlobalVehicleListItem) => row.track_count ?? 'N/A' },
              { key: 'method', header: 'Creation method', render: (row: GlobalVehicleListItem) => row.creation_method || 'N/A' },
              { key: 'seen', header: 'First seen → Last seen', render: (row: GlobalVehicleListItem) => `${row.first_seen_at || 'N/A'} → ${row.last_seen_at || 'N/A'}` },
            ]}
            rows={globalVehiclesQuery.data.items}
            getRowKey={(row) => row.global_vehicle_code}
            onRowClick={(row) => navigate(`/global-vehicles/${encodeURIComponent(row.global_vehicle_code)}`)}
          />
          <Pagination
            page={globalVehiclesQuery.data.page}
            pageSize={globalVehiclesQuery.data.page_size}
            total={globalVehiclesQuery.data.total}
            hasNext={globalVehiclesQuery.data.has_next}
            onPageChange={(page) => updateFilter('page', String(page))}
          />
        </>
      )}
    </div>
  )
}
