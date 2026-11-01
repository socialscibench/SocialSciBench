# R baseline for task 006_matching: MatchIt nearest-neighbour PSM and CEM.
# usage: Rscript bench_r.R data.csv n_cem_covariates out.txt
# CEM uses Stata cem's Sturges cutpoints, passed explicitly (MatchIt's default cutpoints differ).
suppressMessages({library(MatchIt); library(data.table)})
args <- commandArgs(trailingOnly = TRUE)
df <- fread(args[1])
xs <- grep("^x", names(df), value = TRUE)
cx <- xs[seq_len(as.integer(args[2]))]
sturges_cuts <- function(x) { k <- ceiling(log10(length(x)) / log10(2) + 1); seq(min(x), max(x), length.out = k) }

f <- as.formula(paste("d ~", paste(xs, collapse = " + ")))
t0 <- Sys.time()
m <- matchit(f, data = df, method = "nearest", distance = "glm", link = "logit", replace = TRUE, estimand = "ATT")
t1 <- Sys.time()
w <- m$weights
att_psm <- mean(df$y[df$d == 1]) - weighted.mean(df$y[df$d == 0], w[df$d == 0])

fc <- as.formula(paste("d ~", paste(cx, collapse = " + ")))
cuts <- lapply(cx, function(v) sturges_cuts(df[[v]])); names(cuts) <- cx      # not timed
t2 <- Sys.time()
mc <- matchit(fc, data = df, method = "cem", cutpoints = cuts, estimand = "ATT")
t3 <- Sys.time()
wc <- mc$weights
att_cem <- weighted.mean(df$y[df$d == 1], wc[df$d == 1]) - weighted.mean(df$y[df$d == 0], wc[df$d == 0])

out <- file(args[3], "w")
writeLines(sprintf("psm_time %.5f", as.numeric(t1 - t0, units = "secs")), out)
writeLines(sprintf("psm_att %.17g", att_psm), out)
writeLines(sprintf("cem_time %.5f", as.numeric(t3 - t2, units = "secs")), out)
writeLines(sprintf("cem_att %.17g", att_cem), out)
writeLines(sprintf("cem_matched_strata %d", length(unique(mc$subclass[!is.na(mc$subclass)]))), out)
writeLines(sprintf("cem_matched_treated %d", sum(wc > 0 & df$d == 1)), out)
writeLines(sprintf("cem_matched_controls %d", sum(wc > 0 & df$d == 0)), out)
close(out)
