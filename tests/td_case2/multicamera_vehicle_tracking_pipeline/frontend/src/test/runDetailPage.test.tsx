import { describe, expect, it } from 'vitest'
import { screen } from '@testing-library/react'
import RunDetailPage from '../pages/RunDetailPage'
import { renderWithProviders } from './renderWithProviders'

describe('RunDetailPage', () => {
  it('renders run details and summary counts', async () => {
    renderWithProviders(<RunDetailPage />, {
      route: '/runs/RUN_20260724_151402',
      path: '/runs/:runCode',
    })

    expect(await screen.findByText('RUN_20260724_151402')).toBeInTheDocument()
    expect(screen.getByText('Local tracks').closest('.metric-card')).toHaveTextContent('8')
    expect(screen.getByText('Global objects').closest('.metric-card')).toHaveTextContent('7')
  })
})
