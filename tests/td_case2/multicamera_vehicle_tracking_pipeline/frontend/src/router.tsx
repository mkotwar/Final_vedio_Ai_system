import { createBrowserRouter } from 'react-router-dom'
import AppLayout from './components/layout/AppLayout'
import DashboardPage from './pages/DashboardPage'
import RunsPage from './pages/RunsPage'
import RunDetailPage from './pages/RunDetailPage'
import TracksPage from './pages/TracksPage'
import TrackDetailPage from './pages/TrackDetailPage'
import GlobalVehiclesPage from './pages/GlobalVehiclesPage'
import GlobalVehicleDetailPage from './pages/GlobalVehicleDetailPage'
import CrossCameraMatchesPage from './pages/CrossCameraMatchesPage'
import NotFoundPage from './pages/NotFoundPage'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      {
        index: true,
        element: <DashboardPage />,
      },
      {
        path: 'runs',
        element: <RunsPage />,
      },
      {
        path: 'runs/:runCode',
        element: <RunDetailPage />,
      },
      {
        path: 'tracks',
        element: <TracksPage />,
      },
      {
        path: 'tracks/:trackUuid',
        element: <TrackDetailPage />,
      },
      {
        path: 'global-vehicles',
        element: <GlobalVehiclesPage />,
      },
      {
        path: 'global-vehicles/:globalVehicleCode',
        element: <GlobalVehicleDetailPage />,
      },
      {
        path: 'matches',
        element: <CrossCameraMatchesPage />,
      },
      {
        path: '*',
        element: <NotFoundPage />,
      },
    ],
  },
])
