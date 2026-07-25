import { describe, expect, it } from 'vitest'
import { screen } from '@testing-library/react'
import TrackDetailPage from '../pages/TrackDetailPage'
import { renderWithProviders } from './renderWithProviders'

describe('TrackDetailPage', () => {
  it('renders track detail, observations, and reference-only media', async () => {
    renderWithProviders(<TrackDetailPage />, {
      route: '/tracks/RUN_20260724_151402%3ACAM_001%3ATRACK_4',
      path: '/tracks/:trackUuid',
    })
    expect(await screen.findByText('DL8CBF6268')).toBeInTheDocument()
    expect(screen.getAllByText('REFERENCE ONLY')[0]).toBeInTheDocument()
  })
})
