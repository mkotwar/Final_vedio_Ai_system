import { useLocation } from 'react-router-dom'
import ApiHealthIndicator from '../states/ApiHealthIndicator'

const pageTitles: Record<string, { title: string; description: string }> = {
  '/': {
    title: 'Operations Dashboard',
    description: 'Live backend status and recent multicamera processing runs.',
  },
  '/runs': {
    title: 'Processing Runs',
    description: 'Inspect camera coverage, enrichment readiness, and pipeline outputs.',
  },
  '/tracks': {
    title: 'Local Tracks',
    description: 'Filter camera-level tracks by class, colour, OCR, and evidence presence.',
  },
  '/global-vehicles': {
    title: 'Global Vehicles',
    description: 'Review cross-camera vehicle identities and their member tracks.',
  },
  '/matches': {
    title: 'Cross-Camera Matches',
    description: 'Audit deterministic matching decisions, scores, and linked global objects.',
  },
}

export default function Header() {
  const location = useLocation()
  const matched =
    Object.entries(pageTitles).find(([path]) =>
      path === '/'
        ? location.pathname === '/'
        : location.pathname === path || location.pathname.startsWith(`${path}/`),
    )?.[1] || pageTitles['/']

  return (
    <header className="page-header">
      <div>
        <p className="page-header__eyebrow">FastAPI Backend</p>
        <h2 className="page-header__title">{matched.title}</h2>
        <p className="page-header__description">{matched.description}</p>
      </div>
      <ApiHealthIndicator />
    </header>
  )
}
