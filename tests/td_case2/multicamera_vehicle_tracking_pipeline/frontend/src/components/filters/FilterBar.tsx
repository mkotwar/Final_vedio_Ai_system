import type { ReactNode } from 'react'

interface FilterBarProps {
  children: ReactNode
}

export default function FilterBar({ children }: FilterBarProps) {
  return <section className="filter-bar">{children}</section>
}
