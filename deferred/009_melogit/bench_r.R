# R baseline for task 009_melogit: lme4::glmer random-intercept logit with 7-point adaptive
# Gauss-Hermite quadrature (nAGQ = 7, the same number of points as Stata's melogit default; glmer's
# own default is the Laplace approximation, nAGQ = 1).
# usage: Rscript bench_r.R data.csv k out.txt [nAGQ]
suppressMessages({library(lme4); library(data.table)})
args <- commandArgs(trailingOnly = TRUE)
k <- as.integer(args[2])
nagq <- if (length(args) >= 4) as.integer(args[4]) else 7L
df <- fread(args[1])
df[, g := factor(g)]
xs <- paste0("x", seq_len(k))
fml <- as.formula(paste("y ~", paste(xs, collapse = " + "), "+ (1 | g)"))

t0 <- Sys.time()
m <- glmer(fml, data = df, family = binomial, nAGQ = nagq)
t1 <- Sys.time()

b <- fixef(m); se <- sqrt(diag(as.matrix(vcov(m))))
out <- file(args[3], "w")
writeLines(sprintf("time %.5f", as.numeric(t1 - t0, units = "secs")), out)
writeLines(sprintf("N %d", nobs(m)), out)
writeLines(sprintf("ll %.15e", as.numeric(logLik(m))), out)
writeLines(sprintf("var_u %.15e", as.numeric(VarCorr(m)$g[1])), out)
writeLines(sprintf("warnings %d", length(m@optinfo$conv$lme4$messages)), out)
for (x in c(xs, "(Intercept)")) writeLines(sprintf("coef %.15e %.15e", b[[x]], se[[x]]), out)
close(out)
