import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../../api/client'
import type { HealthResponse } from '../../types/common'
import StatusBadge from './StatusBadge'

export default function ApiHealthIndicator() {
  const healthQuery = useQuery({
    queryKey: ['health-indicator'],
    queryFn: () => apiGet<HealthResponse>('/health'),
    retry: 1,
  })

  return (
    <div className="health-indicator">
      <div>
        <p className="health-indicator__label">API health</p>
        <strong className="health-indicator__value">
          {healthQuery.isPending ? 'Checking...' : healthQuery.data?.database || 'Unavailable'}
        </strong>
      </div>
      <StatusBadge
        value={healthQuery.isError ? 'backend_unavailable' : `${healthQuery.data?.status || 'unknown'}`}
      />
    </div>
  )
}
