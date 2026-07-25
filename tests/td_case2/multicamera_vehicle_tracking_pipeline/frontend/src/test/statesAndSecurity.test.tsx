import { describe, expect, it } from 'vitest'
import fs from 'node:fs'
import path from 'node:path'
import { screen } from '@testing-library/react'
import EmptyState from '../components/states/EmptyState'
import NotFoundPage from '../pages/NotFoundPage'
import { renderWithProviders } from './renderWithProviders'

describe('shared states and security', () => {
  it('renders empty and 404 states', () => {
    renderWithProviders(<EmptyState title="Nothing here" description="No rows matched." />, { route: '/', path: '/' })
    expect(screen.getByText('Nothing here')).toBeInTheDocument()
  })

  it('renders the not found route', () => {
    renderWithProviders(<NotFoundPage />, { route: '/missing', path: '/missing' })
    expect(screen.getByText('Page not found')).toBeInTheDocument()
  })

  it('does not include Supabase service role usage in frontend source', () => {
    const srcRoot = path.resolve(__dirname, '..')
    const sourceRoots = ['api', 'components', 'pages', 'types', 'router.tsx', 'App.tsx', 'main.tsx']
    const files = sourceRoots.flatMap((entry) => {
      const fullPath = path.join(srcRoot, entry)
      if (fs.statSync(fullPath).isDirectory()) {
        return (fs.readdirSync(fullPath, { recursive: true }) as string[]).map((file) =>
          path.join(fullPath, file),
        )
      }
      return [fullPath]
    })
    const source = files
      .filter((file) => file.endsWith('.ts') || file.endsWith('.tsx'))
      .map((file) => fs.readFileSync(file, 'utf-8'))
      .join('\n')

    expect(source).not.toContain('SUPABASE_SERVICE_ROLE_KEY')
    expect(source).not.toContain('createClient(')
    expect(source).not.toContain('supabase')
  })
})
