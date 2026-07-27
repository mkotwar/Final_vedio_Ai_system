import { describe, expect, it } from 'vitest'
import { screen } from '@testing-library/react'
import VehicleSearchResultCard from './VehicleSearchResultCard'
import { renderWithProviders } from '../../test/renderWithProviders'
import { vehicleSearchFixture } from '../../test/fixtures'

describe('VehicleSearchResultCard', () => {
  it('opens track detail for a local result', async () => {
    renderWithProviders(<VehicleSearchResultCard result={vehicleSearchFixture.results[1]} />, { route: '/', path: '/' })
    expect(screen.getByRole('link', { name: 'Open details' })).toHaveAttribute('href', '/tracks/RUN_20260725_131944%3ACAM_001%3ATRACK_4')
  })

  it('opens global vehicle detail for a global result', () => {
    renderWithProviders(<VehicleSearchResultCard result={vehicleSearchFixture.results[0]} />, { route: '/', path: '/' })
    expect(screen.getByRole('link', { name: 'Open details' })).toHaveAttribute('href', '/global-vehicles/GVO%3ARUN_20260725_131944%3AFA3FCF9E3ABC')
  })

  it('renders only the vehicle media on search results and keeps plate media hidden', async () => {
    renderWithProviders(<VehicleSearchResultCard result={vehicleSearchFixture.results[0]} />, { route: '/', path: '/' })
    expect(await screen.findByRole('img', { name: 'BEST_VEHICLE_CROP' })).toBeInTheDocument()
    expect(screen.queryByRole('img', { name: 'PLATE_CROP' })).not.toBeInTheDocument()
    expect(screen.queryByText('Plate evidence')).not.toBeInTheDocument()

    renderWithProviders(<VehicleSearchResultCard result={vehicleSearchFixture.results[1]} />, { route: '/', path: '/' })
    expect((await screen.findAllByText(/image unavailable/i)).length).toBeGreaterThan(0)
  })

  it('renders the verification badge, cameras, and match reasons', async () => {
    renderWithProviders(<VehicleSearchResultCard result={vehicleSearchFixture.results[0]} />, { route: '/', path: '/' })
    expect((await screen.findAllByText('VERIFIED')).length).toBeGreaterThan(0)
    expect(screen.getByText('CAM_001, CAM_002')).toBeInTheDocument()
    expect(screen.getByText('exact verified plate')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'CAR' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: /GVO:RUN_/ })).not.toBeInTheDocument()
  })

  it('never promotes a plate crop into the main vehicle image', async () => {
    renderWithProviders(
      <VehicleSearchResultCard
        result={{
          ...vehicleSearchFixture.results[0],
          primary_media: {
            ...vehicleSearchFixture.results[0].primary_plate_media!,
            media_id: 'plate-primary',
          },
          primary_vehicle_media: {
            ...vehicleSearchFixture.results[0].primary_plate_media!,
            media_id: 'plate-primary',
          },
          primary_plate_media: {
            ...vehicleSearchFixture.results[0].primary_plate_media!,
            media_id: 'plate-primary',
          },
        }}
      />,
      { route: '/', path: '/' },
    )

    expect(await screen.findByText('Vehicle image unavailable')).toBeInTheDocument()
    expect(screen.queryByRole('img', { name: 'PLATE_CROP' })).not.toBeInTheDocument()
  })

  it('renders semantic partial plate text in the details section', async () => {
    renderWithProviders(
      <VehicleSearchResultCard
        result={{
          ...vehicleSearchFixture.results[0],
          plate: 'MH12AB1715',
          plate_status: 'PARTIAL',
          plate_result: {
            raw_text: 'MH12AB1715',
            normalized_text: 'MH12AB1715',
            display_text: '...1715',
            status: 'PARTIAL',
            verification_status: 'PARTIAL',
            ocr_confidence: 0.72,
            detector_confidence: 0.7,
            source_media_id: 'media-2',
          },
        }}
      />,
      { route: '/', path: '/' },
    )

    expect((await screen.findAllByText('...1715')).length).toBeGreaterThan(0)
    expect(screen.getAllByText('PARTIAL').length).toBeGreaterThan(0)
  })

  it('prefers persisted plate_result text when legacy plate fields are empty', async () => {
    renderWithProviders(
      <VehicleSearchResultCard
        result={{
          ...vehicleSearchFixture.results[0],
          plate: null,
          plate_status: 'UNKNOWN',
          plate_result: {
            raw_text: 'DL8CBF6268',
            normalized_text: 'DL8CBF6268',
            display_text: 'DL8CBF6268',
            status: 'VERIFIED',
            verification_status: 'VERIFIED',
            ocr_confidence: 0.8,
            detector_confidence: 0.8,
            source_media_id: 'media-2',
          },
        }}
      />,
      { route: '/', path: '/' },
    )

    expect((await screen.findAllByText('DL8CBF6268')).length).toBeGreaterThan(0)
    expect(screen.queryByText('No plate result')).not.toBeInTheDocument()
  })

  it('renders one outer vehicle card without tiled metadata boxes', () => {
    const { container } = renderWithProviders(<VehicleSearchResultCard result={vehicleSearchFixture.results[0]} />, { route: '/', path: '/' })
    expect(container.querySelectorAll('.vehicle-card')).toHaveLength(1)
    expect(container.querySelectorAll('.vehicle-card .metric-card')).toHaveLength(0)
    expect(container.querySelectorAll('.vehicle-card__actions a')).toHaveLength(1)
  })
})
