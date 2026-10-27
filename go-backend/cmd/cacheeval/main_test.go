package main

import (
	"math"
	"testing"
)

func TestCosine(t *testing.T) {
	a := []float32{1, 2, 3}
	if c := cosine(a, a); math.Abs(c-1) > 1e-12 {
		t.Fatalf("identical vectors: %v", c)
	}
	if c := cosine([]float32{1, 0}, []float32{0, 1}); c != 0 {
		t.Fatalf("orthogonal vectors: %v", c)
	}
	if c := cosine([]float32{1, 0}, []float32{-1, 0}); c != -1 {
		t.Fatalf("opposite vectors: %v", c)
	}
	if c := cosine([]float32{3, 4}, []float32{6, 8}); math.Abs(c-1) > 1e-12 {
		t.Fatalf("scale invariance: %v", c)
	}
}
