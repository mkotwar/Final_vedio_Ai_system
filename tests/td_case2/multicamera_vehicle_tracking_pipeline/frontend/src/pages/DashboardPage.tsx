import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../api/client'
import { listRuns } from '../api/runs'
import type { HealthResponse } from '../types/common'
import LoadingState from '../components/states/LoadingState'
import ErrorState from '../components/states/ErrorState'
import EmptyState from '../components/states/EmptyState'
import StatusBadge from '../components/states/StatusBadge'

export default function DashboardPage() {
  const healthQuery = useQuery({
    queryKey: ['dashboard', 'health'],
    queryFn: () => apiGet<HealthResponse>('/health'),
    retry: 1,
  })
  const runsQuery = useQuery({
    queryKey: ['dashboard', 'runs'],
    queryFn: () => listRuns({ page: 1, page_size: 5, sort_by: 'created_at', sort_order: 'desc' }),
    retry: 1,
  })

  if (healthQuery.isPending || runsQuery.isPending) {
    return <LoadingState label="Loading dashboard..." />
  }

  if (runsQuery.isError) {
    return (
      <ErrorState
        title="Dashboard unavailable"
        message="The frontend could not load recent processing runs from FastAPI."
        onRetry={() => {
          void healthQuery.refetch()
          void runsQuery.refetch()
        }}
      />
    )
  }

  const health = healthQuery.data ?? {
    status: healthQuery.isError ? 'unknown' : 'degraded',
    service: 'multicamera-vehicle-api',
    database: healthQuery.isError ? 'unavailable' : 'unknown',
    schema: 'analytics',
  }
  const latestRun = runsQuery.data.items[0]
  const healthHeadline = health.status === 'ok' ? 'FastAPI analytics bridge is online' : 'FastAPI analytics bridge is reachable with warnings'
  const healthDescription = healthQuery.isError
    ? 'Recent processing runs loaded, but the live health probe did not respond.'
    : health.status === 'ok'
      ? 'All frontend data is flowing through the read-only API at '
      : 'The API is responding, but its live analytics database check is currently degraded at '

  return (
    <div className="page-grid">
      <section className="hero-panel">
        <div>
          <p className="hero-panel__eyebrow">Backend</p>
          <h3>{healthHeadline}</h3>
          <p>
            {healthDescription}
            <code>127.0.0.1:8000/api/v1</code>.
          </p>
        </div>
        <div className="hero-panel__stats">
          <div className="metric-card">
            <span>Service</span>
            <strong>{health.service}</strong>
          </div>
          <div className="metric-card">
            <span>Database</span>
            <strong>{health.database}</strong>
          </div>
          <div className="metric-card">
            <span>Schema</span>
            <strong>{health.schema}</strong>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel__header">
          <div>
            <p className="panel__eyebrow">Latest run</p>
            <h3>Most recent processing summary</h3>
          </div>
          {latestRun ? <StatusBadge value={latestRun.status} /> : null}
        </div>
        {latestRun ? (
          <div className="metric-grid">
            <div className="metric-card">
              <span>Run code</span>
              <strong>{latestRun.run_code}</strong>
            </div>
            <div className="metric-card">
              <span>Cameras</span>
              <strong>{latestRun.camera_count}</strong>
            </div>
            <div className="metric-card">
              <span>Tracks</span>
              <strong>{latestRun.track_count}</strong>
            </div>
            <div className="metric-card">
              <span>Global vehicles</span>
              <strong>{latestRun.global_vehicle_count}</strong>
            </div>
            <div className="metric-card">
              <span>Errors</span>
              <strong>{latestRun.processing_error_count}</strong>
            </div>
          </div>
        ) : (
          <EmptyState title="No runs yet" description="The backend did not return any processing runs." />
        )}
      </section>

      <section className="panel">
        <div className="panel__header">
          <div>
            <p className="panel__eyebrow">Recent activity</p>
            <h3>Recent processing runs</h3>
          </div>
          <Link className="button button--secondary" to="/runs">
            View all runs
          </Link>
        </div>
        {runsQuery.data.items.length === 0 ? (
          <EmptyState title="No recent runs" description="Start a processing run to populate the dashboard." />
        ) : (
          <div className="stack-list">
            {runsQuery.data.items.map((run) => (
              <Link key={run.run_code} className="list-row" to={`/runs/${encodeURIComponent(run.run_code)}`}>
                <div>
                  <strong>{run.run_code}</strong>
                  <p>{run.started_at || run.created_at || 'No timestamp available'}</p>
                </div>
                <div className="list-row__meta">
                  <span>{run.track_count} tracks</span>
                  <StatusBadge value={run.status} />
                </div>
              </Link>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
