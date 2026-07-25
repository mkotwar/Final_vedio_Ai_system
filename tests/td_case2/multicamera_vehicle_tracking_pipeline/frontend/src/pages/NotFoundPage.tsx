import { Link } from 'react-router-dom'

export default function NotFoundPage() {
  return (
    <section className="state-card state-card--empty">
      <h3>Page not found</h3>
      <p>The requested route does not exist in the multicamera frontend.</p>
      <Link className="button button--secondary" to="/">
        Return to dashboard
      </Link>
    </section>
  )
}
