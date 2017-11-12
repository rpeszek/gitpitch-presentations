#### JVM, Recursion, and Eta

Robert Peszek, 2017-11

---
![GHC growing Tree](assets/image/ghc-tree.png)

---
#### About Eta

- http://eta-lang.org 
- v0.1 Developer Preview 
- implementation is changing (not documented yet)
- observed recursion behavior: 
   - seems Eta ~ Haskell (*)
   - more space leak sensitive   

Note: seems like Eta leaks space faster but Stack/Heap use patterns seem the same

---
#### JVM and Recursion

- JVM bytecode has no TCO instruction
- poor JIT
- Youtube [Functional Programming is terrible](https://www.youtube.com/watch?v=hzf3hTUKk8U&t=346s)
- Iterop, Performance, Control Flow - pick any two  
_Rich Hickley_

---
#### Stack overflow in Haskell

- very low level stuff (pun)
- do not know much about it 
- want to learn from You

---
#### Stack overflow in Haskell

- laziness changes everything 
- very different (~ opposite to Java/Scala/etc)
  - caused by deeply nested thunks
  - guarded recursion
  - just add lotsa ! :)

---
#### Naive code examples

---
#### Example 1: Mean. Eta (*)
```Haskell
mean :: [Double] -> Double
mean xs = s / fromIntegral n
  where
    (n, s)     = foldl k (0, 0) xs
    k (n, s) x = (n+1, s+x)

-- | Both GHC and Eta stack overflow
bigMean = mean [1..10000000] 
```

Note: example from Real World Haskell

---
#### Example 1: Mean. Eta (*)
```Haskell
{-# LANGUAGE BangPatterns #-}
mean' :: [Double] -> Double
mean' xs = s / fromIntegral n
  where
    (n, s)       = foldl' k (0, 0) xs
    k (!n, !s) x = (n+1, s+x)

-- | Both GHC and Eta work
bigMean' = mean' [1..10000000]
```
Note: stack overflow if fixed by using strict evaluation
foldl' on its own will not fix it

---
#### Example 2: Sum of squares. Scala (*)
(Note not Tail Recursive!):
```Scala
object Recursion {
  def myMap[A,B] (f: A=> B, l: List[A]): List[B] = l match {
    case List()  => List()
    case y :: ys => f(y) :: myMap(f, ys)
  }

  //Stack Overflow
  lazy val bigsum = myMap[Int, Long](sq, List.range(1,100000)).sum  
}
```
Note: I seem to remember that this benchmark has something to do with Wadler

---
#### Example 2: Sum of squares. Eta (*)
```Haskell
myMap :: (a -> b) -> [a] -> [b]
myMap _ [] = []
myMap f (x:xs) = f x : myMap f xs

-- works! with heavy space use
bigSum = sum $ myMap (^2) [1..1000000] 
```
Note: using myMap to avoid built-in optimizations. I have tried controlling implementation of sum as well.
In GHC this costs 270MB/40ms, Eta is slower and I did not profile space use. Eta will OutOfMemory error at 10x the size.

---
#### Example 2b: Sum of squares TC. Scala 
```Scala
object Recursion {
  @tailrec
  def myMapAux[A,B] (f: A=> B, l: List[A], acc:List[B]): List[B] = {
     l match {
       case List() => acc
       case x :: xs => myMapAux(f, xs, f(x) :: acc)
     }
  }
  def myMapTr[A,B] (f: A=> B, l: List[A]): List[B] = myMapAux[A,B](f, l, List())

  //works (but is UGLY)
  lazy val bigsum2 = myMapTr[Int, Long](sq, List.range(1,1000000)).sum 
}
```

---
#### Example 2b: Sum of squares TC. Eta (*)
Just so we do not look at Scala
```Haskell
myMapAux :: [b] -> (a -> b) -> [a]  -> [b]
myMapAux acc _ [] = acc
myMapAux acc f (x:xs) = myMapAux ((f x) : acc) f xs

myMap' :: (a -> b) -> [a] -> [b]
myMap' = myMapAux []

bigSum' = sum $ myMap' (^2) [1..1000000]
```
Note: this slide is so we do not look at Scala. This code change benefits Scala but not Haskell/Eta.

---
#### Example 2c: Sum of squares with Vector. Eta
(Highly lambda optimized, Notice 100x bigger)
```Haskell
import qualified Data.Vector as V

sq x = x * x
-- | blasting fast, small space in GHC
--   OutOfMemory in Eta 
bigSumVec = let xs = V.enumFromTo  (100000000 :: Int64)
            in V.sum $ V.map sq $ xs
```

---
#### Example 3: Mutual Recursion. Eta
```Haskell
isEven :: Integer -> Bool
isEven 0 = True
isEven i = isOdd $ i - 1

isOdd :: Integer -> Bool
isOdd 0 = False
isOdd i = isEven $ i - 1

evenMM = isEven 1000000  -- Works!!
```

---
#### Example 3: Mutual Recursion. Scala
```Scala
  // Forget it 
  // @tailrec does not work on mutually recursive code
```

---
#### Some References
- http://eta-lang.org 
- http://book.realworldhaskell.org/read/profiling-and-optimization.html
- https://www.well-typed.com/blog/2014/05/understanding-the-stack/
