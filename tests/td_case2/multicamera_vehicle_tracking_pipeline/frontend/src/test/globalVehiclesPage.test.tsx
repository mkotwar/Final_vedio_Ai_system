import { describe, expect, it } from 'vitest'
import { screen } from '@testing-library/react'
import GlobalVehiclesPage from '../pages/GlobalVehiclesPage'
import GlobalVehicleDetailPage from '../pages/GlobalVehicleDetailPage'
import { renderWithProviders } from './renderWithProviders'

describe('Global vehicles views', () => {
  it('renders the global vehicles list', async () => {
    renderWithProviders(<GlobalVehiclesPage />, { route: '/global-vehicles', path: '/global-vehicles' })
    expect(await screen.findByText('GVO:RUN_20260724_151402:943BD1FE7C62')).toBeInTheDocument()
  })

  it('renders the verified global vehicle detail', async () => {
    renderWithProviders(<GlobalVehicleDetailPage />, {
      route: '/global-vehicles/GVO%3ARUN_20260724_151402%3A943BD1FE7C62',
      path: '/global-vehicles/:globalVehicleCode',
    })
    expect(await screen.findByText('DL8CBF6268')).toBeInTheDocument()
    expect(screen.getByText('CAR')).toBeInTheDocument()
    expect(screen.getByText('GREY')).toBeInTheDocument()
    expect(screen.getAllByText('RUN_20260724_151402:CAM_001:TRACK_4')).toHaveLength(2)
    expect(screen.getAllByText('RUN_20260724_151402:CAM_002:TRACK_4')).toHaveLength(2)
  })
})
