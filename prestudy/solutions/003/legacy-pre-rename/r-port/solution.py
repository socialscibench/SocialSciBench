"""Task 008 competitor: the repository's R port of reclink2 (data.table + stringi), in-process via rpy2.

The R code is baselines/008_recordlinkage/bench_r.R (the "R port" column of reports/008_recordlinkage.md):
f32, bigram1 and the timed body from `m[, i := .I]` to `rbindlist(res)`, unchanged except that it is
wrapped in a function returning the pairs, and data.table uses one thread. Each column is copied into R
as one joined string split in R (faster than rpy2's per-string conversion); the copies are inside the
timed call. COMPARATOR selects the study's Rcpp comparator instead (solutions/008/r-rcpp).
"""
import os

import numpy as np

os.environ.setdefault("R_HOME", "/Library/Frameworks/R.framework/Resources")
import rpy2.robjects as ro  # noqa: E402

R_LIBS = os.environ.get("SOCIALSCIBENCH_R_LIBS",
                        "/path/to/repro-socialscibench/research/2026-10-14/R-library")
COMPARATOR = "stringi"
SEP = "\x1f"
VARS = ["name", "addr", "city", "zip", "state"]

ro.r(f'.libPaths(c("{R_LIBS}", .libPaths())); suppressMessages({{library(data.table); library(stringi)}})')
ro.r("setDTthreads(1)")
ro.r(r'''
vars <- c("name", "addr", "city", "zip", "state")
wm <- c(10, 8, 5, 5, 5); wnm <- c(2, 3, 4, 4, 4); req <- "state"
minscore <- 0.6; minbigram <- 0.6; biscale <- .5 / (1 - minbigram)^(1 / 3)

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

split_col <- function(x, n) if (n == 0) character(0) else strsplit(x, "\x1f", fixed = TRUE)[[1]]

link <- function(mcols, ucols, n_m, n_u) {
  m <- as.data.table(setNames(lapply(mcols, split_col, n = n_m), vars))
  u <- as.data.table(setNames(lapply(ucols, split_col, n = n_u), vars))
  # --- bench_r.R timed body ---
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
  # --- end of timed body ---
  list(i = as.numeric(res$i) - 1, j = as.numeric(res$j) - 1, score = as.numeric(res$score),
       perfect = nrow(perf))
}
''')
if COMPARATOR == "rcpp":
    # The study's compiled comparator (research/2026-10-14/tasks007008/bench_r.R), built at import.
    ro.r('suppressMessages(library(Rcpp))')
    ro.r(r'''cppFunction(code='
  NumericVector bigram_compiled(CharacterVector ms, CharacterVector us) {
    int n=ms.size(); NumericVector out(n);
    for(int r=0;r<n;r++) {
      std::string m=as<std::string>(ms[r]), s=as<std::string>(us[r]);
      int lm=m.size(), ls=s.size(), cnt=0, wink=0; double g=0;
      if(lm>2) {
        for(int k=0;k<lm-1;k++) if(s.find(m.substr(k,2))!=std::string::npos) cnt++;
        g=(float)(cnt*2.0/(ls+lm-2));
      }
      if(lm<3 || ls<3) {
        if(m.find(s)!=std::string::npos || s.find(m)!=std::string::npos)
          g=(float)(std::min(lm,ls)*1.0/(lm+ls-1));
        else g=0;
      }
      for(int k=1;k<=4;k++) if(s.substr(0,k)==m.substr(0,k)) wink++;
      g=(float)(g+wink*(1-g)/10);
      if(m==s) g=1;
      if(ls==0) g=0;
      out[r]=g;
    }
    return out;
  }')''')
    ro.r("bigram1 <- bigram_compiled")
_link = ro.r["link"]


def _column(values):
    """One R string: the values each followed by SEP (strsplit keeps empty values except a final
    empty piece, which the terminator provides)."""
    a = np.asarray(values, dtype=str)
    joined = SEP.join(a.tolist()) + SEP
    if joined.count(SEP) != len(a):
        raise ValueError("values must not contain the separator")
    return ro.StrVector([joined])


def run(master, using, vars, wmatch, wnomatch, required, minscore=0.6, minbigram=0.6):
    if (list(vars), list(wmatch), list(wnomatch), list(required), minscore, minbigram) != (
            VARS, [10, 8, 5, 5, 5], [2, 3, 4, 4, 4], ["state"], 0.6, 0.6):
        raise ValueError("the R port hard-codes the task's fields, weights, required(state) and thresholds")
    n_m, n_u = len(master[VARS[0]]), len(using[VARS[0]])
    out = _link(ro.ListVector({v: _column(master[v]) for v in VARS}),
                ro.ListVector({v: _column(using[v]) for v in VARS}), n_m, n_u)
    res = dict(zip(out.names, out))
    return (np.asarray(res["i"], dtype=np.int64), np.asarray(res["j"], dtype=np.int64),
            np.asarray(res["score"], dtype=np.float64), int(res["perfect"][0]))
