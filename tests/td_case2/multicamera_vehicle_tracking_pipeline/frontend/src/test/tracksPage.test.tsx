import { describe, expect, it } from 'vitest'
import { screen } from '@testing-library/react'
import TracksPage from '../pages/TracksPage'
import { renderWithProviders } from './renderWithProviders'

describe('TracksPage', () => {
  it('shows the run-code required empty state', () => {
    renderWithProviders(<TracksPage />, { route: '/tracks', path: '/tracks' })
    expect(screen.getByText('Run code required')).toBeInTheDocument()
  })

  it('renders track filters and data', async () => {
    renderWithProviders(<TracksPage />, {
      route: '/tracks?run_code=RUN_20260724_151402',
      path: '/tracks',
    })
    expect(await screen.findByText('DL8CBF6268')).toBeInTheDocument()
    expect(screen.getByText('GREY')).toBeInTheDocument()
  })
})
