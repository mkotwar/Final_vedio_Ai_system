import { describe, expect, it } from 'vitest'
import { screen } from '@testing-library/react'
import CrossCameraMatchesPage from '../pages/CrossCameraMatchesPage'
import { renderWithProviders } from './renderWithProviders'

describe('CrossCameraMatchesPage', () => {
  it('renders confirmed match statuses', async () => {
    renderWithProviders(<CrossCameraMatchesPage />, { route: '/matches', path: '/matches' })
    expect(await screen.findByText('RUN_20260724_151402:CAM_001:TRACK_4')).toBeInTheDocument()
    expect(screen.getByText('CONFIRMED')).toBeInTheDocument()
  })
})
