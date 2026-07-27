import { describe, expect, it, vi } from 'vitest'
import { screen } from '@testing-library/react'
import TrackDetailPage from './TrackDetailPage'
import { renderWithProviders } from '../test/renderWithProviders'
import { trackDetailFixture, observationsFixture, mediaFixture } from '../test/fixtures'

describe('TrackDetailPage', () => {
  it('renders available evidence images and track details', async () => {
    renderWithProviders(<TrackDetailPage />, {
      route: '/tracks/RUN_20260724_151402%3ACAM_001%3ATRACK_4',
      path: '/tracks/:trackUuid',
    })

    expect((await screen.findAllByText('DL8CBF6268')).length).toBeGreaterThan(0)
    expect(screen.getByText('Linked to global vehicle')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'GVO:RUN_20260724_151402:943BD1FE7C62' })).toHaveAttribute(
      'href',
      '/global-vehicles/GVO%3ARUN_20260724_151402%3A943BD1FE7C62',
    )
    expect(screen.getByRole('link', { name: 'Open Global Vehicle' })).toHaveAttribute(
      'href',
      '/global-vehicles/GVO%3ARUN_20260724_151402%3A943BD1FE7C62',
    )
    expect(screen.getByRole('img', { name: 'BEST_VEHICLE_CROP' })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'PLATE_CROP' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'CAR' })).toBeInTheDocument()
    expect(screen.getByText('RUN_20260724_151402:CAM_001:TRACK_4')).toBeInTheDocument()
    expect(screen.getByText('Class stabilization')).toBeInTheDocument()
    expect(screen.getByText('Raw class')).toBeInTheDocument()
    expect(screen.getAllByText('BUS').length).toBeGreaterThan(0)
    expect(screen.queryByText('Reference only')).not.toBeInTheDocument()
  })

  it('shows not linked when the API returns an unlinked membership', async () => {
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(typeof input === 'string' ? input : input.toString())
      const path = decodeURIComponent(url.pathname)

      if (path.endsWith('/tracks/RUN_20260724_151402:CAM_001:TRACK_4')) {
        return new Response(
          JSON.stringify({
            ...trackDetailFixture,
            global_membership: { linked: false },
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        )
      }
      if (path.endsWith('/tracks/RUN_20260724_151402:CAM_001:TRACK_4/observations')) {
        return new Response(JSON.stringify(observationsFixture), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      }
      if (path.endsWith('/tracks/RUN_20260724_151402:CAM_001:TRACK_4/media')) {
        return new Response(JSON.stringify(mediaFixture), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      }

      return new Response(
        JSON.stringify({ error: { code: 'NOT_FOUND', message: 'Not found.', details: null } }),
        { status: 404, headers: { 'Content-Type': 'application/json' } },
      )
    })

    renderWithProviders(<TrackDetailPage />, {
      route: '/tracks/RUN_20260724_151402%3ACAM_001%3ATRACK_4',
      path: '/tracks/:trackUuid',
    })

    expect(await screen.findByText('Not linked')).toBeInTheDocument()
    expect(screen.queryByText('Open Global Vehicle')).not.toBeInTheDocument()
  })
})
