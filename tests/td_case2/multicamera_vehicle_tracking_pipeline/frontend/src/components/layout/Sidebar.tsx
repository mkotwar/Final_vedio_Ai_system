import { NavLink } from 'react-router-dom'

const navItems = [
  { to: '/', label: 'Dashboard' },
  { to: '/runs', label: 'Runs' },
  { to: '/tracks', label: 'Tracks' },
  { to: '/global-vehicles', label: 'Global Vehicles' },
  { to: '/matches', label: 'Cross-Camera Matches' },
  { to: '/search', label: 'Vehicle Search' },
]

export default function Sidebar() {
  return (
    <aside className="sidebar" aria-label="Primary navigation">
      <div className="sidebar__brand">
        <p className="sidebar__eyebrow">Multicamera</p>
        <h1 className="sidebar__title">Vehicle Intelligence</h1>
        <p className="sidebar__subtitle">Read-only investigation console</p>
      </div>
      <nav className="sidebar__nav">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            className={({ isActive }) => `sidebar__link${isActive ? ' sidebar__link--active' : ''}`}
            to={item.to}
            end={item.to === '/'}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}
