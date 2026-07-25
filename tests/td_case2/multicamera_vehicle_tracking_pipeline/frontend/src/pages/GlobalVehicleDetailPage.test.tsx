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

    expect(await screen.findByText('DL8CBF6268')).toBeInTheDocument()
    expect(screen.getAllByRole('img', { name: 'BEST_VEHICLE_CROP' })).toHaveLength(2)
  })
})
