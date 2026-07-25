interface PaginationProps {
  page: number
  pageSize: number
  total: number
  hasNext: boolean
  onPageChange: (page: number) => void
}

export default function Pagination({
  page,
  pageSize,
  total,
  hasNext,
  onPageChange,
}: PaginationProps) {
  const start = total === 0 ? 0 : (page - 1) * pageSize + 1
  const end = Math.min(total, page * pageSize)

  return (
    <div className="pagination">
      <p className="pagination__summary">
        Showing {start}-{end} of {total}
      </p>
      <div className="pagination__actions">
        <button className="button button--secondary" disabled={page <= 1} onClick={() => onPageChange(page - 1)} type="button">
          Previous
        </button>
        <span className="pagination__page">Page {page}</span>
        <button className="button button--secondary" disabled={!hasNext} onClick={() => onPageChange(page + 1)} type="button">
          Next
        </button>
      </div>
    </div>
  )
}
