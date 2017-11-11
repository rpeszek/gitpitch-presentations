#### JVM, Recursion, and Eta

Robert Peszek, 2017-11
---
#### JVM and Recursion

- Bytecode has no TCO instruction
- [Functional Programming is terrible youtube](https://www.youtube.com/watch?v=hzf3hTUKk8U&t=346s)
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
  - guarded recursion
  - just add lotsa ! :)

---
![GHC growing Tree](assets/image/ghc-tree.png)

---
#### About Eta

- http://eta-lang.org 
- v0.1 Developer Preview
- implementation is changing (and not documented yet)
- recursion behavior: 
   - seems Eta ~ Haskell 
   - more space leak sensitive

---
#### Naive code examples


---
#### Example 1. Scala
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

---
#### Example 1. Eta
```Haskell
myMap :: (a -> b) -> [a] -> [b]
myMap _ [] = []
myMap f (x:xs) = f x : myMap f xs

bigSum = sum $ myMap (^2) [1..1000000] -- works!
```

---
#### Example 1b. Scala 
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
#### Code Example 1b. Eta 
Just so we do not look at Scala
```Haskell
myMapAux :: [b] -> (a -> b) -> [a]  -> [b]
myMapAux acc _ [] = acc
myMapAux acc f (x:xs) = myMapAux ((f x) : acc) f xs

myMap' :: (a -> b) -> [a] -> [b]
myMap' = myMapAux []

bigSum' = foldl' (+) 0 $ myMap' (^2) [1..1000000]
```
---
#### Code Example 2. Eta
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
#### Code Example 2. Scala
```Scala
  // Forget it 
  // @tailrec does not work on mutually recursive code
```
---
#### Code Example 3. Eta
```Haskell
mean :: [Double] -> Double
mean xs = s / fromIntegral n
  where
    (n, s)     = foldl k (0, 0) xs
    k (n, s) x = (n+1, s+x)

-- | Both GHC and Eta stack overflow
bigMean = mean [1..10000000] 
```
---
#### Code Example 3. Eta
```Haskell
mean' :: [Double] -> Double
mean' xs = s / fromIntegral n
  where
    (n, s)       = foldl' k (0, 0) xs
    k (!n, !s) x = (n+1, s+x)

-- | Neither GHC nor Eta stack overflows
bigMean' = mean' [1..100000000]
```
