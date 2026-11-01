# R port for task 003_reclink (formerly 008_recordlinkage): reclink2's algorithm (perfect matches, or-blocking with
# required variables, bigram1 comparator, match/non-match weights, best candidate per master record)
# with data.table joins and stringi. No CRAN package implements reclink's scoring, so this is a port;
# like reclink2 it rounds every intermediate score to single precision (Stata float).
# usage: Rscript bench_r.R master.csv using.csv out.txt [threads]
suppressMessages({library(data.table); library(stringi)})
args <- commandArgs(trailingOnly = TRUE)
setDTthreads(if (length(args) >= 4) as.integer(args[4]) else 12)
vars <- c("name", "addr", "city", "zip", "state")
wm <- c(10, 8, 5, 5, 5); wnm <- c(2, 3, 4, 4, 4); req <- "state"
minscore <- 0.6; minbigram <- 0.6; biscale <- .5 / (1 - minbigram)^(1 / 3)
m <- fread(args[1], colClasses = "character", na.strings = NULL)
u <- fread(args[2], colClasses = "character", na.strings = NULL)

f32 <- function(x) readBin(writeBin(x, raw(), size = 4), "double", size = 4, n = length(x))

bigram1 <- function(ms, us) {                       # master values, using values (equal length)
  lm <- nchar(ms); ls <- nchar(us)
  cnt <- numeric(length(ms))
  for (k in seq_len(max(lm, 2) - 1)) {
    idx <- which(lm > k)
    cnt[idx] <- cnt[idx] + stri_detect_fixed(us[idx], substr(ms[idx], k, k + 1))
  }
  g <- ifelse(lm > 2, f32(cnt * 2 / (ls + lm - 2)), 0)
  short <- which(lm < 3 | ls < 3)
  if (length(short)) {
    cont <- ls[short] == 0 | lm[short] == 0              # an empty string is contained in any string
    ne <- which(!cont)
    cont[ne] <- stri_detect_fixed(ms[short][ne], us[short][ne]) | stri_detect_fixed(us[short][ne], ms[short][ne])
    g[short] <- ifelse(cont, f32(pmin(lm[short], ls[short]) / (lm[short] + ls[short] - 1)), 0)
  }
  wink <- 0
  for (i in 1:4) wink <- wink + (substr(us, 1, i) == substr(ms, 1, i))
  g <- f32(g + wink * (1 - g) / 10)
  g[ms == us] <- 1
  g[ls == 0] <- 0
  g
}

t0 <- Sys.time()
m[, i := .I]; u[, j := .I]
perf <- merge(m, u, by = vars)[, .(i, j)]
mr <- m[!i %in% perf$i]
ur <- u[!j %in% perf$j]
orblock <- setdiff(vars, req)
res <- list(perf[, .(i, j, score = 1)])
chunk <- 20000L
for (s in seq(1, max(nrow(mr), 1), by = chunk)) {
  mc <- mr[s:min(s + chunk - 1, nrow(mr))]
  if (!nrow(mc)) break
  p <- unique(rbindlist(lapply(orblock, function(v)
    mc[get(v) != "", c(v, req, "i"), with = FALSE][ur[, c(v, req, "j"), with = FALSE], on = c(v, req),
                                                   nomatch = NULL, allow.cartesian = TRUE][, .(i, j)])))
  if (!nrow(p)) next
  M <- numeric(nrow(p)); NM <- numeric(nrow(p))
  for (k in seq_along(vars)) {
    v <- vars[k]
    va <- mc[[v]][match(p$i, mc$i)]; vb <- ur[[v]][match(p$j, ur$j)]
    me <- va == ""; ue <- vb == ""
    if (v %in% req) {
      M <- ifelse(me, M, f32(M + wm[k]))
    } else {
      key <- data.table(a = va, b = vb)
      pr <- unique(key)                                 # score each distinct pair of values once
      pr[, bi := bigram1(a, b)]
      bi <- pr[key, on = c("a", "b"), bi]
      bonus <- ifelse(bi > minbigram, biscale * pmax(bi - minbigram, 0)^(1 / 3), 0)
      M <- ifelse(me, M, f32(M + wm[k] * (.5 * bi^2 + bonus)))
      pen <- ifelse(bi < minbigram - .2, 1, ifelse(bi < minbigram, 1 - bi^2, (1 - bi)^2))
      NM <- ifelse(me | ue, NM, f32(NM + wnm[k] * pen))
    }
    NM <- ifelse(me != ue, f32(NM + wnm[k] * 0.3), NM)
  }
  p[, score := f32(M / (M + NM))]
  p <- p[, .SD[score == max(score) & score >= minscore], by = i]
  res[[length(res) + 1]] <- p
}
res <- rbindlist(res, use.names = TRUE)
t1 <- Sys.time()

out <- file(args[3], "w")
writeLines(sprintf("time %.5f", as.numeric(t1 - t0, units = "secs")), out)
writeLines(sprintf("pairs %d", nrow(res)), out)
writeLines(sprintf("matched_master %d", uniqueN(res$i)), out)
writeLines(sprintf("perfect %d", nrow(perf)), out)
writeLines(sprintf("score_sum %.17g", sum(res$score)), out)
writeLines(sprintf("id_master_sum %.17g", sum(as.numeric(res$i))), out)
writeLines(sprintf("id_using_sum %.17g", sum(as.numeric(res$j))), out)
close(out)
