# R baseline for task 005_merge: data.table::merge(..., all = TRUE), an outer merge like Stata's default.
# usage: Rscript bench_r.R master.csv using.csv keys(comma-separated) key_type(numeric|string) out.txt [threads]
suppressMessages(library(data.table))
args <- commandArgs(trailingOnly = TRUE)
keys <- strsplit(args[3], ",")[[1]]
setDTthreads(if (length(args) >= 6) as.integer(args[6]) else 12)
cc <- if (args[4] == "string") setNames(as.list(rep("character", length(keys))), keys) else NULL
cls <- if (is.null(cc)) NULL else list(character = keys)
x <- fread(args[1], colClasses = cls)
y <- fread(args[2], colClasses = cls)
x[, in_x := 1L]                                 # flags to rebuild Stata's _merge (not timed)
y[, in_y := 1L]

t0 <- Sys.time()
m <- merge(x, y, by = keys, all = TRUE)
t1 <- Sys.time()

code <- ifelse(is.na(m$in_y), 1L, ifelse(is.na(m$in_x), 2L, 3L))
num <- setdiff(names(m), c(keys, "in_x", "in_y"))
out <- file(args[5], "w")
writeLines(sprintf("time %.5f", as.numeric(t1 - t0, units = "secs")), out)
writeLines(sprintf("rows %d", nrow(m)), out)
for (k in 1:3) writeLines(sprintf("merge %d %d", k, sum(code == k)), out)
for (c in num) writeLines(sprintf("col %s %d %.17g", c, sum(!is.na(m[[c]])), sum(m[[c]], na.rm = TRUE)), out)
close(out)
