import { describe, expect, it } from 'vitest'
import { screen } from '@testing-library/react'
import GlobalVehicleDetailPage from './GlobalVehicleDetailPage'
import { renderWithProviders } from '../test/renderWithProviders'

describe('GlobalVehicleDetailPage', () => {
  it('shows evidence for both member tracks when available', async () => {
    renderWithProviders(<GlobalVehicleDetailPage />, {
      route: '/global-vehicles/GVO%3ARUN_20260724_151402%3A943BD1FE7C62',
      path: '/global-vehicles/:globalVehicleCode',
    })

    expect((await screen.findAllByText('DL8CBF6268')).length).toBeGreaterThan(0)
    expect(screen.getAllByRole('img', { name: 'BEST_VEHICLE_CROP' })).toHaveLength(3)
    expect(screen.getAllByRole('img', { name: 'PLATE_CROP' })).toHaveLength(3)
    expect(screen.getAllByRole('button', { name: /Open .* preview/ })).toHaveLength(6)
  })
})
