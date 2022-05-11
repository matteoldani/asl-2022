input_sizes = c(100, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 3500)

for (r in input_sizes) {
  m <- matrix(runif(r*r), r, r);
  NMF::nmf.stop.iteration(1000)
  res <- NMF::nmf(data.frame(m), 3);
  print(r);
  print(res@runtime);
}
