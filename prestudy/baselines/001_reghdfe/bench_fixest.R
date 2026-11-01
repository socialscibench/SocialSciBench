# R fixest baseline for task 001_reghdfe_ols_cluster: feols with absorbed fixed effects and
# clustered standard errors (fixest defaults otherwise).
# usage: Rscript bench_fixest.R data.csv n_regressors out.txt [threads]
suppressMessages({library(fixest); library(data.table)})
args <- commandArgs(trailingOnly = TRUE)
k <- as.integer(args[2])
setFixest_nthreads(if (length(args) >= 4) as.integer(args[4]) else 12)
df <- fread(args[1])
xs <- paste0("x", seq_len(k))
fes <- grep("^fe[0-9]+$", names(df), value = TRUE)
fml <- as.formula(paste("y ~", paste(xs, collapse = " + "), "|", paste(fes, collapse = " + ")))

t0 <- Sys.time()
m <- feols(fml, data = df, cluster = ~cl)
t1 <- Sys.time()

out <- file(args[3], "w")
writeLines(sprintf("time %.5f", as.numeric(t1 - t0, units = "secs")), out)
writeLines(sprintf("N %d", nobs(m)), out)
for (x in xs) writeLines(sprintf("coef %.15e %.15e", coef(m)[[x]], se(m)[[x]]), out)
close(out)
