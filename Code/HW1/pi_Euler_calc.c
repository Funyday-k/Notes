#include <stdio.h>
#include <math.h>

int main() {
  int n = 100000000;
  double pi = 0.0;

  for (int i = 1; i < n; i++) {
    pi += 1 / pow(i , 2);
  }

  pi *= 6;

  printf("PI = %.15lf\n", sqrt(pi));

  int a;

  scanf("%d",&a);

  printf("%d",a);

  return 0;
}