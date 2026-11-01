# R baseline for task 011_logithdfe_ame: fixest::feglm + marginaleffects::avg_slopes.
# usage: Rscript bench_fixest.R data.csv n_continuous out.txt [threads]
# The data have columns y, d (0/1 treatment), x1..xk, fe1..feG (fixed effects) and cl (cluster).
# marginaleffects cannot compute standard errors of AMEs after feglm with fixed effects
# (it stops with an error), so AMEs are computed with vcov = FALSE: point estimates only.
suppressMessages({library(fixest); library(marginaleffects); library(data.table)})
args <- commandArgs(trailingOnly = TRUE)
k <- as.integer(args[2])
setFixest_nthreads(if (length(args) >= 4) as.integer(args[4]) else 12)
df <- fread(args[1])
df[, d := as.logical(d)]                       # so the AME of d is a discrete change, as in Stata
xs <- c("d", paste0("x", seq_len(k)))
fes <- grep("^fe[0-9]+$", names(df), value = TRUE)
fml <- as.formula(paste("y ~", paste(xs, collapse = " + "), "|", paste(fes, collapse = " + ")))

t0 <- Sys.time()
m <- feglm(fml, data = df, family = binomial, cluster = ~cl,
           ssc = ssc(K.adj = FALSE, G.adj = TRUE))   # matches Stata: q = G / (G - 1)
t1 <- Sys.time()
a <- avg_slopes(m, variables = xs, vcov = FALSE)
t2 <- Sys.time()

out <- file(args[3], "w")
writeLines(sprintf("time_feglm %.5f", as.numeric(t1 - t0, units = "secs")), out)
writeLines(sprintf("time_ame %.5f", as.numeric(t2 - t1, units = "secs")), out)
writeLines(sprintf("N %d", nobs(m)), out)
writeLines(sprintf("iterations %d", m$iterations), out)
for (x in c("dTRUE", xs[-1])) writeLines(sprintf("coef %.15e %.15e", coef(m)[[x]], se(m)[[x]]), out)
for (x in xs) writeLines(sprintf("ame %.15e", a$estimate[a$term == x]), out)
close(out)
