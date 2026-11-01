# R baseline for task 007_fuzzymatch: stringdist's cosine similarity on 2-grams (the same score as
# Stata matchit's default), all pairs, computed block by block and filtered at the threshold.
# usage: Rscript bench_r.R master.txt using.txt threshold out.txt [threads]
suppressMessages(library(stringdist))
args <- commandArgs(trailingOnly = TRUE)
a <- readLines(args[1]); b <- readLines(args[2]); thr <- as.numeric(args[3])
threads <- if (length(args) >= 5) as.integer(args[5]) else 12
block <- max(1L, as.integer(2e7 %/% length(b)))

t0 <- Sys.time()
I <- list(); J <- list(); S <- list(); k <- 0
for (s in seq(1, length(a), by = block)) {
  e <- min(s + block - 1, length(a))
  M <- stringsimmatrix(a[s:e], b, method = "cosine", q = 2, nthread = threads)
  idx <- which(M >= thr, arr.ind = TRUE)
  k <- k + 1
  I[[k]] <- idx[, 1] + s - 1; J[[k]] <- idx[, 2]; S[[k]] <- M[idx]
}
I <- unlist(I); J <- unlist(J); S <- unlist(S)
t1 <- Sys.time()

out <- file(args[4], "w")
writeLines(sprintf("time %.5f", as.numeric(t1 - t0, units = "secs")), out)
writeLines(sprintf("pairs %d", length(S)), out)
writeLines(sprintf("score_sum %.17g", sum(S)), out)
writeLines(sprintf("id_master_sum %.17g", sum(as.numeric(I))), out)
writeLines(sprintf("id_using_sum %.17g", sum(as.numeric(J))), out)
close(out)
