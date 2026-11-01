# R baseline for task 010_larger_than_memory: each recipe the usual way in R, one year at a time where
# that saves memory: haven::read_dta (only the needed columns), data.table, haven::write_dta.
# usage: R --no-echo --no-restore --file=bench_r.R --args recipe output.dta timefile file1.dta file2.dta ...
suppressMessages({library(haven); library(data.table)})
args <- commandArgs(trailingOnly = TRUE)
recipe <- args[1]; out <- args[2]; timefile <- args[3]; files <- args[-(1:3)]
setDTthreads(12)
from <- as.Date("2015-01-01")

t0 <- Sys.time()
if (recipe == "extract") {
  cols <- c("user_id", "rcid", "startdate", "enddate", "seniority", "salary", "role_k1500_v2", "state", "country")
  d <- rbindlist(lapply(files, function(f) {
    x <- setDT(read_dta(f, col_select = all_of(cols)))
    x[country == "United States" & (is.na(startdate) | startdate >= from)][, country := NULL]
  }))
  d[, year := year(startdate)]
  # codes in byte order, as Stata's encode (method = "radix" sorts in the C locale)
  roles <- sort(unique(d$role_k1500_v2[d$role_k1500_v2 != ""]), method = "radix")
  states <- sort(unique(d$state[d$state != ""]), method = "radix")
  d[, role := labelled(match(role_k1500_v2, roles), setNames(seq_along(roles), roles))]
  d[, state := labelled(match(state, states), setNames(seq_along(states), states))]
  d <- d[, .(user_id, rcid, startdate, enddate, year, seniority, salary, role, state)]
  write_dta(d, out)
}
if (recipe == "aggregate") {
  cols <- c("rcid", "startdate", "seniority", "salary", "country")
  parts <- rbindlist(lapply(files, function(f) {
    x <- setDT(read_dta(f, col_select = all_of(cols)))
    x <- x[country == "United States" & (is.na(startdate) | startdate >= from)]
    x[, quarter := (year(startdate) - 1960L) * 4L + (month(startdate) - 1L) %/% 3L]
    x[, .(n_positions = .N, n_salary = sum(!is.na(salary)), sum_salary = sum(salary, na.rm = TRUE),
          n_senior = sum(!is.na(seniority) & seniority >= 4)), by = .(rcid, quarter)]
  }))
  d <- parts[, lapply(.SD, sum), by = .(rcid, quarter)]
  d[, mean_salary := fifelse(n_salary > 0, sum_salary / n_salary, NA_real_)]
  setorder(d, rcid, quarter, na.last = TRUE)
  write_dta(d[, .(rcid, quarter, n_positions, n_salary, mean_salary, n_senior)], out)
}
if (recipe == "append") {
  d <- rbindlist(lapply(files, function(f) setDT(read_dta(f))), use.names = TRUE, fill = TRUE)
  write_dta(d, out)
}
t1 <- Sys.time()
writeLines(sprintf("time %.5f", as.numeric(t1 - t0, units = "secs")), timefile)
