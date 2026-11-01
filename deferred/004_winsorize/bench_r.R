# R baseline for task 004_winsorize: data.table, winsorizing with quantile(type = 2),
# which matches Stata's percentile definition (R's default, type 7, does not).
# usage: Rscript bench_r.R data.csv n_vars has_group(0/1) p_lo p_hi out.txt [threads]
suppressMessages(library(data.table))
args <- commandArgs(trailingOnly = TRUE)
k <- as.integer(args[2]); has_group <- args[3] == "1"
p <- as.numeric(args[4:5]) / 100
setDTthreads(if (length(args) >= 7) as.integer(args[7]) else 12)
dt <- fread(args[1])
vars <- paste0("v", seq_len(k))
orig <- copy(dt[, ..vars])                     # for the checks below (not timed)

winsorize <- function(x) {
  q <- quantile(x, p, type = 2, na.rm = TRUE, names = FALSE)
  pmin(pmax(x, q[1]), q[2])                    # missing values stay missing
}

t0 <- Sys.time()
if (has_group) {
  dt[, (vars) := lapply(.SD, winsorize), by = g, .SDcols = vars]
} else {
  dt[, (vars) := lapply(.SD, winsorize), .SDcols = vars]
}
t1 <- Sys.time()

out <- file(args[6], "w")
writeLines(sprintf("time %.5f", as.numeric(t1 - t0, units = "secs")), out)
for (v in vars) {
  w <- dt[[v]]; x <- orig[[v]]; m <- !is.na(w)
  writeLines(sprintf("check %d %.17g %.17g %.17g %.17g %d", sum(m), sum(w[m]), sum(w[m]^2), min(w[m]), max(w[m]),
                     sum(w[m] != x[m])), out)
}
close(out)
