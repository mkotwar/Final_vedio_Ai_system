import { useMemo } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { listRuns } from '../api/runs'
import type { RunListItem } from '../types/run'
import DataTable from '../components/tables/DataTable'
import Pagination from '../components/tables/Pagination'
import FilterBar from '../components/filters/FilterBar'
import LoadingState from '../components/states/LoadingState'
import ErrorState from '../components/states/ErrorState'
import EmptyState from '../components/states/EmptyState'
import StatusBadge from '../components/states/StatusBadge'

export default function RunsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const navigate = useNavigate()

  const filters = useMemo(
    () => ({
      page: Number(searchParams.get('page') || '1'),
      page_size: Number(searchParams.get('page_size') || '10'),
      status: searchParams.get('status') || undefined,
      run_code: searchParams.get('run_code') || undefined,
      sort_by: searchParams.get('sort_by') || 'created_at',
      sort_order: searchParams.get('sort_order') || 'desc',
    }),
    [searchParams],
  )

  const runsQuery = useQuery({
    queryKey: ['runs', filters],
    queryFn: () => listRuns(filters),
    placeholderData: keepPreviousData,
  })

  function updateFilter(name: string, value: string) {
    const next = new URLSearchParams(searchParams)
    if (name === 'page') {
      next.set(name, value)
    } else if (value) {
      next.set(name, value)
    } else {
      next.delete(name)
    }
    if (name !== 'page') {
      next.set('page', '1')
    }
    setSearchParams(next)
  }

  if (runsQuery.isPending && !runsQuery.data) {
    return <LoadingState label="Loading processing runs..." />
  }

  if (runsQuery.isError) {
    return (
      <ErrorState
        title="Runs unavailable"
        message="The frontend could not load processing runs from FastAPI."
        onRetry={() => void runsQuery.refetch()}
      />
    )
  }

  const columns = [
    {
      key: 'run_code',
      header: 'Run code',
      render: (row: RunListItem) => (
        <div>
          <strong>{row.run_code}</strong>
          <p className="table-subtext">{row.created_at || 'No creation timestamp'}</p>
        </div>
      ),
    },
    {
      key: 'status',
      header: 'Status',
      render: (row: RunListItem) => <StatusBadge value={row.status} />,
    },
    {
      key: 'started_at',
      header: 'Started',
      render: (row: RunListItem) => row.started_at || 'N/A',
    },
    {
      key: 'completed_at',
      header: 'Completed',
      render: (row: RunListItem) => row.completed_at || 'N/A',
    },
    {
      key: 'camera_count',
      header: 'Cameras',
      render: (row: RunListItem) => row.camera_count,
    },
    {
      key: 'track_count',
      header: 'Local tracks',
      render: (row: RunListItem) => row.track_count,
    },
    {
      key: 'global_vehicle_count',
      header: 'Global vehicles',
      render: (row: RunListItem) => row.global_vehicle_count,
    },
    {
      key: 'processing_error_count',
      header: 'Errors',
      render: (row: RunListItem) => row.processing_error_count,
    },
  ]

  return (
    <div className="page-stack">
      <FilterBar>
        <label>
          <span>Status</span>
          <select value={filters.status || ''} onChange={(event) => updateFilter('status', event.target.value)}>
            <option value="">All</option>
            <option value="COMPLETED">Completed</option>
            <option value="RUNNING">Running</option>
            <option value="FAILED">Failed</option>
          </select>
        </label>
        <label>
          <span>Run code</span>
          <input
            placeholder="Search run code"
            value={filters.run_code || ''}
            onChange={(event) => updateFilter('run_code', event.target.value)}
          />
        </label>
        <label>
          <span>Sort by</span>
          <select value={filters.sort_by} onChange={(event) => updateFilter('sort_by', event.target.value)}>
            <option value="created_at">Created</option>
            <option value="started_at">Started</option>
            <option value="completed_at">Completed</option>
            <option value="run_code">Run code</option>
            <option value="status">Status</option>
          </select>
        </label>
        <label>
          <span>Order</span>
          <select value={filters.sort_order} onChange={(event) => updateFilter('sort_order', event.target.value)}>
            <option value="desc">Descending</option>
            <option value="asc">Ascending</option>
          </select>
        </label>
        <button className="button button--secondary" onClick={() => void runsQuery.refetch()} type="button">
          Refresh
        </button>
      </FilterBar>

      {runsQuery.data.items.length === 0 ? (
        <EmptyState title="No runs found" description="Try clearing filters or waiting for a new processing run." />
      ) : (
        <>
          <DataTable
            columns={columns}
            rows={runsQuery.data.items}
            getRowKey={(row) => row.run_code}
            onRowClick={(row) => navigate(`/runs/${encodeURIComponent(row.run_code)}`)}
          />
          <Pagination
            page={runsQuery.data.page}
            pageSize={runsQuery.data.page_size}
            total={runsQuery.data.total}
            hasNext={runsQuery.data.has_next}
            onPageChange={(page) => updateFilter('page', String(page))}
          />
        </>
      )}
    </div>
  )
}
