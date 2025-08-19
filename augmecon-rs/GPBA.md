```
European Journal of Operational Research 306 (2023) 286–
```
Contents lists available at ScienceDirect

## European Journal of Operational Research

journal homepage: [http://www.elsevier.com/locate/ejor](http://www.elsevier.com/locate/ejor)

## Decision Support

# New −constraint methods for multi-objective integer linear

## programming: A Pareto front representation approach

## Mariana Mesquita-Cunha

```
∗
```
## , José Rui Figueira, Ana Paula Barbosa-Póvoa

_CEGIST - Centre for Management Studies, Instituto Superior Técnico, Universidade de Lisboa, Portugal_

## a r t i c l e i n f o

_Article history:_

Received 26 July 2021

Accepted 24 July 2022

Available online 29 July 2022

_Keywords:_

Multiple objective programming

Integer linear programming

Generation methods

Representation methods

## a b s t r a c t

```
Dealing with multi-objective problems by using generation methods has some interesting advantages
```
```
since it provides the decision-maker with the complete information about the set of non-dominated cri-
terion vectors (Pareto front) and a clear overview of the different trade-offs of the problem. However,
```
```
providing many solutions to the decision-maker may also be overwhelming. As an alternative approach,
showing a representative set of the Pareto front may be advantageous. Choosing such a representative set
```
```
is by itself also a multi-objective problem that must consider the number of alternatives to present, the
uniformity, and/or the coverage of the representation, to guarantee its quality. This paper proposes three
```
```
algorithms for the representation problem for multi-objective integer linear programming problems with
```
```
two or more objective functions, each one of them dealing with each dimension of the problem (cardi-
```
## nality, coverage, and uniformity). Such algorithms are all based on the -constraint approach. In addition,

```
the paper also presents strategies to overcome poor estimations of the Pareto front bounds. The algo-
rithms were tested on the ability to efficiently generate the whole Pareto front or a representation of
```
```
it. The uniformity and cardinality algorithms proved to be very efficient both on binary and on integer
problems, being amongst the best in the literature. Both coverage and uniformity algorithms provide good
```
```
quality representations on their targeted objective, while the cardinality algorithm appears to be the most
flexible, privileging uniformity for lower cardinality representations and coverage on higher cardinality.
```
```
©2022 The Authors. Published by Elsevier B.V.
```
```
This is an open access article under the CC BY-NC-ND license
( http://creativecommons.org/licenses/by-nc-nd/4.0/ )
```
**1. Introduction**

Most real-world problems, although being multi-objective by

their very nature, are frequently modelled as single objective prob-

lems mainly due to the lack of multi-objective tools and/or the

computational complexity of solving multi-objective models. How-

ever, the manipulation of objectives in order to amalgamate them

into a single objective function tends to oversimplify the problem

and the conflicting nature among objectives. Indeed, a single objec-

tive model may not represent appropriately the decision-maker’s

(DM) real preferences and objectives ( Branke, Deb, Miettinen, &

Slowi nski, ́ 2008 ). The aforementioned disadvantages together with

the rapid rise of computational performance, both in terms of

hardware and software, are making multi-objective models, and in

particular Multi-Objective Optimization (MOO), more favoured ap-

proaches. Nevertheless, multi-objective integer and mixed integer

linear programming (MOILP and MOMILP), which arise in many

real-world applications (such as logistics and production prob-

```
∗Corresponding author.
```
```
E-mail address: mariana.cunha@tecnico.ulisboa.pt (M. Mesquita-Cunha).
```
lems), have received less attention when compared to binary and

continuous problems ( Alves & Clímaco, 2007 ).

Depending on the stage at which the DM is involved to as-

certain his/her preferences, multi-objective methods can be clas-

sified into _a priori_ methods, interactive methods, and _a posteri-_

_ori_ /generation methods. In _a priori_ methods, the DM states her/his

preferences at the begining and the problem is then solved by ag-

gregating the objective functions into a single one. This approach,

not only carries many of the disadvantages presented above for

the single objective models but also, most of the time, in complex

functions, the DM struggles to provide parameters as, for example,

the weights for each objective. In interactive methods, the DM it-

eratively states and adjusts preferences based on the results previ-

ously obtained. The drawback with this approach is two-folded: (1)

it may take a long time for a satisfactory trade-off between objec-

tives to be found; and (2) the DM, by never seeing the Pareto front,

also never clearly understands the relation between objectives and

their impact on the solutions. _A posteriori_ or generation methods,

which involve the DM only after finding the Pareto front, are the

most advantageous. This approach clearly allows for defining the

trade-off between all objectives, providing the DM with a full view

https://doi.org/10.1016/j.ejor.2022.07.

0377-2217/© 2022 The Authors. Published by Elsevier B.V. This is an open access article under the CC BY-NC-ND license ( [http://creativecommons.org/licenses/by-nc-nd/4.0/](http://creativecommons.org/licenses/by-nc-nd/4.0/) )


of the problem and all the tools for making a better-informed de-

cision. However, not only have they become more computationally

heavy, but they also risk overwhelming the DM with too many so-

lutions to analyse.

The _a posteriori_ methods frequently require significant compu-

tational effort and time to compute the efficient/non-dominated

set. Hence, apart from usually small and linear problems, the non-

dominated set is not practical to compute in a reasonable amount

of time. As a result, there are two classes of generation methods,

one that aims to determine the set of non-dominated criterion vec-

tors (the so-called Pareto front), and the other which focuses on

obtaining a set of points representative of the non-dominated set

without being overwhelming for the DM ( Alves & Clímaco, 2007;

Kidd, Lusby, & Larsen, 2020 ). To address the latter, the represen-

tation problem must be considered, which, in itself is a multi-

objective problem with three objective functions, namely (1) car-

dinality, the number of alternatives presented to the DM; (2) cov-

erage, how well the Pareto front is being represented by the set

of solutions; and (3) uniformity, how well those alternatives are

spread through the Pareto front ( Sayın, 20 0 0 ).

In this work, we propose three multi-objective _a posteriori_ al-

gorithms to solve MOILP/ MOMILP with more than two objectives

with strategies to overcome poor quality approximations for the

bounds of the Pareto front, that can be used to generate the whole

Pareto front or to compute a representation of it. To this end, the

algorithms we put forward tackle each of the three objective func-

tions of the representation problem.

Using a generation method on a MOILP/MOMILP is often more

challenging than on a multi-objective linear program since the

former has a non-convex feasible region, which implies that

there may exist unsupported non-dominated criterion vectors.

Chalmet, Lemonidis, and Elzinga (1986) proposed a modified

weighted-sum method to produce, apart from the set of sup-

ported non-dominated criterion vectors, the set of unsupported

non-dominated criterion vectors. This was achieved by adding con-

straints to the problem that bound the values of the objective func-

tions. Although all non-dominated points can be obtained with a

full parametrization of the weights added to each objective func-

tion, it can lead to an extensive and computationally demanding

optimization problem.

Several other authors also proposed approaches based on a se-

quential reduction of the feasible region ( Klein & Hannan, 1982;

Masin & Bukchin, 2007; Sylva & Crema, 20 04; 20 07 ). Klein and

Hannan (1982) propose a reduction of the feasible region by se-

quentially adding constraints that eliminate points dominated by

the previously found non-dominated criterion vector. This strategy

finds points spaced by at least a fixed amount (provided by a pa-

rameter), for each objective. However, it may skip interesting so-

lutions from the DM’s preferences point of view, since only one of

the objectives is considered as the objective function to be opti-

mized. Sylva and Crema (2004) presented a variation of Klein and

Hannan (1982) ’s method where all the objectives are included in

the objective function by using weighting parameters. Sylva and

Crema (2007) build upon the Sylva and Crema (2004) method

by determining, at each iteration, the weights that maximize the

infinity-norm distance to the set dominated by the previously

found solutions. Consequently, uniformly spaced points are found.

Masin and Bukchin (2007) proposed a very similar approach. All

of these methods share the same main drawback, namely the fact

that the problem size increases whenever a non-dominated crite-

rion vectors is found, since the new constraints and, in some cases,

new variables have to be added to the model. Hence, updating the

search region whenever a new non-dominated criterion vectors is

obtained is very important ( Ceyhan, Köksalan, & Lokman, 2019;

Dächert, Klamroth, Lacour, & Vanderpooten, 2017; Klamroth, La-

cour, & Vanderpooten, 2015 ). Although focused on computing the

whole Pareto front, Klamroth et al. (2015) present two strategies

to incrementally update the problem’s search region on this type

of generation methods, redundancy elimination and redundancy

avoidance, considering the geometrical properties of the search

region and decreasing the number of redundant computations.

Dächert et al. (2017) use the neighbourhood relation between lo-

cal bounds of the search region to update it, providing a signifi-

cantly more efficient strategy than the ones proposed in Klamroth

et al. (2015). Ceyhan et al. (2019) and Do gan, ̆ Lokman, and Kök-

salan (2022) propose algorithms to generate representations that

guarantee a predefined coverage error. To that end, Ceyhan et al.

(2019) present a first algorithm (SBA) that partitions the search

space into subsets and searches for the worst-represented points

in each of these subsets, retaining the one with maximum cov-

erage gap; a second algorithm (TDA), which eliminates from the

search region the space around each non-dominated criterion vec-

tor corresponding to the desired coverage gap; and a third algo-

rithm (SPA) which requires a desired cardinality from the DM and,

by approximating the Pareto front to an _L_ (^) _p_ surface, predicts the
location of the points providing the best coverage representation.
However, while the second algorithm, TDA, requires the DM to pro-
vide a desired coverage threshold value, which is often very diffi-
cult, the third algorithm, SPA, only provides better results than the
first algorithm on the lowest cardinalities, falling short in terms
of coverage gap on higher cardinalities. Do gan ̆ et al. (2022) take
as a baseline algorithm TDA and combine it with Dächert et al.
(2017) search region update strategy, proposing algorithm TSGA.
However, since these algorithms are sensitive to the scalarization
model’s weight parameters, Do ̆gan et al. (2022) determine the
model’s weights by fitting an _L_ (^) _p_ surface and using an hyperplane
tangent to the surface on the point that is at minimum Cheby-
shev distance from the ideal point. TSGA proved more efficient
than SBA and TDA, proposed by Ceyhan et al. (2019). Kidd et al.
(2020) also present a scalarization algorithm targeted for generat-
ing representations for bi-objective problems. To that end, an in-
sertion method based on Voronoi cuts to partition the search re-
gion was developed. This method was proven to achieve simulta-
neously good coverage and uniformity for a given cardinality. The
algorithm continues to insert solutions until a desired cardinality
level is reached.
One of the most popular methods for solving _a posteriori_ MOILP
problems is the -constraint method. For constraining each objec-
tive, the method resorts to a virtual grid spaced, for each objective,
by making use of an parameter. Laumanns, Thiele, and Zitzler
(2006) applied the -constraint method as a generation method
by dividing the objective space, after each iteration, using the val-
ues of the previous computed solution to create search boxes in
the objective space. Additionally, after each new solution is found,
a lexicographic optimization is performed to ensure getting non-
dominated criterion vectors. Kirlik and Sayın (2014) developed a
method based on the Laumanns et al. (2006) works, which makes
use of a two stage model in order to guarantee non-dominated cri-
terion vectors. It is a search approach based on the construction of
rectangles, as in Laumanns et al. (2006) , but it also removes rect-
angles in which no non-dominated criterion vector can be found.
This approach makes the search less exhaustive when compared
to the Laumanns et al. (2006) algorithm. Mavrotas (2009) adapted
the original -constraint method by introducing slack variables in
equality constraints and incorporating them in the objective func-
tion weighted by a factor which includes the range of each objec-
tive. To choose the vector, the range of each objective is divided
into evenly spaced intervals generating a uniform grid. However,
this approach produces several redundant points. To overcome this,
Mavrotas and Florios (2013) extended the Mavrotas (2009) method
by exploring the values of the slack variables in order to skip
redundant iterations, improving computational efficiency. Zhang


and Reimann (2014) addressed the requirement to have the

true nadir values, which are often difficult to compute (see

Alves & Costa, 2009; Ehrgott & Ryan, 2003; Kirlik & Sayın, 2015 ).

The proposed methodology skips the redundant iterations in a

way that does not require more computations when using ap-

proximated nadir values. The disadvantage, however, is that it can

only be used for the computation of the whole Pareto front and

not for the generation of a representation. Ozlen, Burton, and

MacRae (2014) proposed an adaption to a previously developed -

constraint based algorithm, proposed in Özlen and Azizo glu ̆ (2009) ,

integrating information provided by the computed points, which

allowed to skip redundant iterations. However, comparative anal-

ysis, reported in Zhang and Reimann (2014) , proved the method-

ology by Zhang and Reimann (2014) to be more efficient than

the proposed by Ozlen et al. (2014). Nikas, Fountoulakis, Forouli,

and Doukas (2020) also address Mavrotas and Florios (2013) draw-

backs but these authors present an algorithm that requires an ar-

ray with as many entries as the integer size of the objective func-

tions’ ranges, which can generate memory issues. Furthermore, the

presented results report dominated criterion vectors, requiring the

obtained points to be filtered. Despite the fact that both Mavrotas

and Florios (2013) and Nikas et al. (2020) allow for the computa-

tion of Pareto front representations, their quality is not addressed

on both works.

Eusébio, Figueira, and Ehrgott (2014) address the representation

problem using the -constraint method for bi-objective problems.

Eusébio et al. (2014) propose two algorithms, one for coverage and

another for uniformity. The former is an insertion based method:

at each iteration, the algorithm searches for a criterion vector be-

tween the two most distant in the representation. The latter suc-

cessively adds criterion vectors spaced by a predetermined step to

the representation. Both methods terminate when the desired level

of coverage and uniformity are met.

In this work, we take as a baseline Mavrotas and Florios

(2013) and combine it with the strategies presented in Zhang

and Reimann (2014) , which allow, without having to calculate the

whole Pareto front, to pass over redundant iterations and also per-

mit to overcome the need of computing the true nadir points. In

this way, we can address the aforementioned drawbacks of both

works, while keeping their advantages. Furthermore, we develop

three search strategy algorithms, one for coverage, another for uni-

formity, and the third one for cardinality, addressing each dimen-

sion of the Pareto front representation problem. The first two al-

gorithms are based on the work by Eusébio et al. (2014) and are

extended for MOILP problems with more than two objectives. The

latter refines the virtual grid, introduced by Mavrotas (2009) , af-

ter finding each solution, trying to verify the cardinality level by

focusing the search on the feasible region. This work, by the very

nature of its model formulation and strategy to look for new crite-

rion vectors in the Pareto front, is independent of both the num-

ber of non-dominated solutions found and the objective functions

ranges.

The remainder of this paper is organized as follows.

Section 2 introduces the mathematical background, namely

the type of problem addressed in this paper, the model used

and the representation problem. Section 3 presents the proposed

methodology, both the generic algorithm and the three proposed

search strategies, as well as an illustrative example for each one

of them. Section 4 provides the computational results for the

strategies presented in Section 3 , both the ones for computing the

Pareto front as well as those for the representation problem. At

last, Section 5 presents some concluding remarks and future work

lines are put forward.

**2. Mathematical background**

This section presents the main concepts, definitions, and no-

tation on multi-objective optimization, the −constrain approach,

and some fundamental concepts related to the representation of

the whole set of criterion vectors.

_2.1. The multi-objective integer programming problem_

Consider the following multi-objective integer linear program-

ming model.

```
max z 1 ( x ) = ( c
1
)

x
```
```
.
.
.
```
```
.
.
.
```
max _z_ (^) _k_ ( _x_ ) = ( _c
k_
)

_x_
.
.
.
.
.
.
max _z_ (^) _p_ ( _x_ ) = ( _c
p_
)

_x_
subject to: _x_ ∈ _X_.
(P1)
where _x_ = ( _x_ 1 ,... , _x_ (^) _j_ ,... , _x_ (^) _n_ ) is an _n_ −vector of non-negative and
integer _decision variables_ , ( _c
k_
)

= ( _c
k_
1
, ... , _c
k
j_
,... , _c
k
n_ )^ is^ an^ _n_^ −row^
vector composed of the _coefficients_ of the decision variables in the
_objective functions_ , _k_ = 1 ,... , _p_ (we assume these coefficients are
integer values or can be converted into integer), and _X_ is the _fea-
sible region_ in the _decision space_ , Z
_n_
0 +
(the set of non-negative inte-
gers). Let _Z_ = _z_ ( _X_ ) denote the image of the _feasible region_ accord-
ing to the objective functions. The set _Z_ is called the _feasible region_
in the _objective space_ , i.e., Z
_p_
(the set of integers) along with the
order relation imposed by the objective functions in this set. Fur-
thermore, assume the feasible region in the decision space is both
bounded and not empty.
Problem 1 can be presented in a more compact form as follows.
“ max ” _z_ ( _x_ ) = _Cx_
subject to: _x_ ∈ _X_.
(P2)
where “max ”means that all functions are to be maximized, _z_ ( _x_ ) =
( _z_ 1 ( _x_ ) , ... , _z_ (^) _k_ ( _x_ ) ,... , _z_ (^) _p_ ( _x_ )) is the vector of the _p_ objective func-
tions, and _C_ is an _p_ × _n_ matrix, each row being composed of the
coefficients of each objective function.
**Definition 1.** (Dominance) Let _z_
′
and _z_
′′
denote two criterion vec-
tors, or points, in the objective space. Then, vector _z_
′
_dominates z_
′′
,
iff _z_
′
≥ _z_
′′
with _z_
′
 = _z_
′′
(i.e., _z_
′
_k_
 _z_
′′
_k_
, with at least a strict inequality,
for _k_ = 1 ,... , _p_ ).
**Definition 2.** (Strict dominance) Let _z_
′
and _z_
′′
denote two criterion
vectors, or points, in the objective space. Then, vector _z_
′
_strictly
dominates z_
′′
, iff _z_
′
> _z_
′′
with _z_
′
 = _z_
′′
(i.e., _z_
′
_k_
> _z_
′′
_k_
for _k_ = 1 ,... , _p_ ).
Let _z_
∗
denote a feasible criterion vector, i.e., _z_
∗
∈ _Z_. This vector
is a _weakly non-dominated criterion vector_ if and only if there is
no other _z_ ∈ _Z_ such that _z_ strictly dominates _z_ ∗. One weakly non-
dominated criterion vector is more interesting than another if its
performance is better in at least one objective function. If there is
no vector _z_ ∈ _Z_ , such that _z_ dominates _z_
∗
, vector _z_
∗
is designated as
a _non-dominated criterion vector_. Whenever _z_
∗
can be obtained as
a weighted-sum of the _p_ objective functions with strictly positive
weighting factors, _z_
∗
is called a _supported criterion vector_. Other-
wise, _z_
∗
is said to be an _unsupported criterion vector_. _N_ ( _Z_ ) denotes
the set of non-dominated criterion vectors, also called the _Pareto
front_.
The same kind of concepts can be applied in the decision space.
A feasible solution _x_ ∗ is called an _efficient solution_ if and only
if we cannot find another _x_ ∈ _X_ such that _Cx_ ≥ _Cx_
∗
and _Cx_  = _Cx_
∗
.
A feasible solution _x_ ∗is a _weakly efficient solution_ if and only if


we cannot find another _x_ ∈ _X_ such that _Cx_ > _Cx_
∗

. Furthermore, for

any solution _x_ ∗corresponding to an inverse image of a _supported_

( _unsupported_ ) criterion vector _z_
∗
is called an efficient _supported_

( _unsupported_ ) solution. Finally, _E_ ( _X_ ) denotes the set of all effi-

cient solutions. For more details about these definitions see Ehrgott

(2005) andSteuer (1986).

_2.2. The_ − _constraint approach_

This subsection is devoted to presenting an approach for iden-

tifying the Pareto front, based on the resolution of a sequence of

problems of the following form.

```
max
x
```
{

_z_ (^) _q_ ( _x_ )
}
subject to: _z_ (^) _k_ ( _x_ )   _k_ , _k_ = 1 ,... , _p_ , _k_  = _q
x_ ∈ _X_.
(P3)
**Theorem 1.** _( Haimes, Lasdon, & Wismer , 1971) If x_
∗
_is an optimal
solution of Problem 3 , for some q , then x_
∗
_is a weakly efficient solution
of Problem 1._
**Proof.** See Chankong and Haines (2008) , Ehrgott (2005) , or Haimes
et al. (1971). 
Algorithm 1 can be implemented for solving a sequence of
Problem 3 versions. This algorithm requires as inputs Problem
2 data, namely matrix _C_ and the feasible region _X_ , and a small
enough strictly positive parameter value, η. It provides as output
_N_ ( _Z_ ) , the Pareto front or, if unable to compute all efficient solution,
part of it. This algorithm makes use of four internal procedures:
**Algorithm 1:** Computing the Pareto front, ( _N_ ( _Z_ )).
**1 Input:** _C_ , _X_ , η;
**2 Output:** _N_ ( _Z_ ) ;
**3** _N_ ˆ^ ( _Z_ ) ← {} ;
**4** _z_
∗
← _Ideal_ ( _P_ ) ;
**5** _z
nad_
← _ApproxNadir_ ( _P_ ) ;
**6** Select _q_ = 1 and build Problem P3;
**7**  _h_ ← _z
nad
h_
, for _h_ = 2 ,... , _p_ ;
**8 while** ( _z_ ˆ 2 < _z_
∗
2
_and X_  = ∅ ) **do
9 while** ( _z_ ˆ 3 < _z_
∗
3
_and X_  = ∅ ) **do
10**...
**11 while** ( _z_ ˆ (^) _p_ − 1 < _z_
∗
_p_ − 1
_and X_  = ∅ ) **do
12 while** ( _z_ ˆ^ _p_ < _z_
∗
_p and_^ _X_^ ^ =^ ∅^ )^ **do**^
**13** ˆ _z_ ← _Solv e_ ( _P_ 3 ) ;
**14** _N_ ˆ^ ( _Z_ ) ← _N_ ˆ^ ( _Z_ ) ∪ { _z_ ˆ } ;
**15**  _p_ ← _z_ ˆ (^) _p_ + η;
**16**  _p_ − 1 ←  _p_ − 1 + η;
**17**  _p_ ← _z
nad
p_ ;^
**18**...
**19**  3 ←  3 + η;
**20**  4 ← _z
nad_
4
;
**21**  2 ←  2 + η;
**22**  3 ← _z
nad_
3
;
**23** _N_ ( _Z_ ) ← _F ilter_ ( _N_ ˆ^ ( _Z_ )) ;
**24 return** ( _N_ ( _Z_ )) ;

1. _Ideal_ ( _P_ ) : which computes the ideal point, _z_
    ∗
       =

```
( z
∗
1
,... , z
∗
k
,... , z
∗
p )^ ,^ through^ an^ individual^ maximization^ of^
each objective function.
```
2. _ApproxNadir_ ( _P_ ) : which computes the nadir point or an approx-

```
imation of it, z nad^ = ( z nad^
1
,... , z nad^
k
```
```
,... , z nad^
p
).
```
3. _Solv e_ ( _P_
    
       ) : which makes use of an integer linear program-

ming solver to solve _P_ , and provides the criterion vector _z_ ̄ =

( _z_ ̄ 1 , ... , _z_ ̄ (^) _k_ , ... , _z_ ̄ (^) _p_ ).

4. _F ilter_ ( _N_ ˆ^ ( _Z_ )) : which makes the filtering of an auxiliary set,

_N_ ˆ^ ( _Z_ ) , which may contain weakly non-dominated criterion vec-

tors, and provides as output the Pareto front, _N_ ( _Z_ ).

5. From the first two calculations we identify the ranges of possi-

ble values, for each objective function, which are bounded from

```
below by z
nad
k
and from above by z
∗
k
, for k = 1 ,... , p.
```
6. In each _while_ loop the vector bounding the objective func-

tions is updated for the following iteration.

The −constraint approach consists of solving a sequence of

versions of Problem 3 by successive increments or decrements

of parameters  _k_ , for _k_ = 1 ,... , _p_ with _k_  = _q_. Any basic algo-

rithm designed for implementing this approach (as for example

Algorithm 1 ) suffers from three major drawbacks, independently

of the solver used for optimizing a variation of Problem 3 :

1. Due to the model structure, unnecessary versions of Problem

3 are solved, since it may find some weakly efficient solutions

with no interest (i.e., disposable weakly efficient solutions). This

is due to the shape of the model objective function and appears

as a conclusion of Theorem 1.

2. Additionally, due to the model structure, a large amount of

processing time may be required to compute the Pareto front

( Ehrgott & Ryan, 2003 ). This is due to the lack of flexibility

of the constraints ( Ehrgott & Ruzika, 2008; Ehrgott & Ryan,

2003 ).

3. Finally, due to the difficulty of adjusting the vector, weakly

efficient solutions of interest may be missed ( Laumanns et al.,

2006 ).

The third drawback was addressed by Laumanns et al. (2006).

These authors show in detail the serious issues related to the _a pri-_

_ori_ technical adjustment of the vector, and propose an adaptive

based algorithm with an interesting scheme for determining the

adequate values for the vector, when sequentially solving each

variation of Problem 3.

The second drawback was reported and first addressed by

Ehrgott and Ryan (2003). These authors proposed an elastic tech-

nique for changing the nature of type constraints. As a conse-

quence, the following model was proposed (some improved ver-

sions of this model can be seen in Ehrgott & Ruzika 2008 ).

```
max
x , e −, e +^
```
{

_z_ (^) _q_ ( _x_ ) −
_p_
∑
_k_ = 1 , _k_  = _q
p_ (^) _k e_
−
_k_
}
subject to: _z_ (^) _k_ ( _x_ ) + _e_
−
_k_
− _e_
+
_k_
=  _k_ , _k_ = 1 ,... , _p_ , _k_  = _q
e_
−
_k_
, _e_
+
_k_
 0 , _k_ = 1 ,... , _p_ , _k_  = _q
x_ ∈ _X_.
(P4)
The following result did not change the nature of the output,
persisting the first drawback.
**Theorem 2.** _( Ehrgott & Ryan (2003) ) If p_ (^) _k_ > 0 _, for k_ = 1 ,... , _pwith
k_  = _p, and_ ( _x_
∗
, _e_
−∗
, _e_
+ ∗
) _is an optimal solution of Problem 4 , for some
q , then x_
∗
_is a weakly efficient solution of Problem 1._
**Proof.** See Ehrgott and Ryan (2003) , or Ehrgott and Ruzika
(2008) 
The first drawback was addressed by Mavrotas (2009) guar-
anteeing the efficiency of the obtained solution with the pro-
posal of a new model based on the introduction of slack vari-
ables. Mavrotas and Florios (2013) proposed a new improvement
for multi-objective integer linear programming which makes a
slight change on the objective function. This change will not alter
the theoretical results presented in Mavrotas (2009). The improved


model can be stated as follows.

```
max
x , s
```
{

_z_ (^) _q_ ( _x_ ) + ρ
_p_
∑
_k_ = 1 , _k_  = _q_
10
_k_ − 1 _s_^ _k
r_ (^) _k_
}
subject to: _z_ (^) _k_ ( _x_ ) − _s_ (^) _k_ =  _k_ , _k_ = 1 ,... , _p_ , _k_  = _q
s_ (^) _k_  0 , _k_ = 1 ,... , _p_ , _k_  = _q
x_ ∈ _X_.
(P5)
where _r_ (^) _k_ is the width of the range values for each objective func-
tion, _k_ = 1 ,... , _p_ , _k_  = _q_.
The result of solving a version of the previous problem obeys to
the following theoretical result.
**Theorem 3.** _( Mavrotas 2009 ) If_ ( _x_ ∗, _s_ ∗) _is an optimal solution of
Problem 5 , for some q and for_ ρ _corresponding to a sufficiently small
number (usually between_ 10
− 3
_and_ 10
− 6
_), then x_
∗
_is an efficient so-
lution of Problem 1._
**Proof.** See Mavrotas (2009). 
Problem 5 will be used in our algorithms.
_2.3. On the representation of the Pareto front_
A discrete representation of the Pareto front, _R_ ( _N_ ) , is a finite
subset of the Pareto front, _N_ ( _Z_ ). Many studies have focused on the
quality of representations, in terms of how well the subset cap-
tures the characteristics of the full set (see Audet, Bigeon, Cartier,
Le Digabel, & Salomon, 2021; Faulkenberg & Wiecek, 2010; Sayın,
20 0 0 ). These studies proposed dimensions of interest in their eval-
uation of the representation. Sayın (20 0 0) proposes three criteria:

1. _Coverage_ : How well does the representation cover all regions of

the objective space _Z_ included in _N_ ( _Z_ ).

2. _Uniformity_ : How diverse and equally spaced are the points in

the representation, i.e., how spread are the points (criterion

vectors) included in the representation.

3. _Cardinality_ : The number of criterion vectors considered in the

representation, ( _R_ ( _N_ )).

Coverage and uniformity are dependent on the distance be-

tween points in the representation. There are multiple distance

metrics, the most common being as follows:

_d_ ( _z_ , _z_

```
′
) =
```
```
⎧
⎨
```
⎩

```
(
| z 1 − z
′
1
|
t
```
+ ···+ | _z_ (^) _k_ − _z_
′
_k_
|
_t_
+ ···+ | _z_ (^) _p_ − _z_
′
_p_ |^
_t_
) 1 / _t_
, _t_  1 ,
max
(
| _z_ 1 − _z_
′
1
| ,... , | _z_ (^) _k_ − _z_
′
_k_
| ,... , | _z_ (^) _p_ − _z_
′
_p_ |^
)
, _t_ = ∞.
where, if _t_ = 1 the distance metric used is the Manhattan distance,
if _t_ = 2 it corresponds to the Euclidean distance, and if _t_ = ∞ the
Chebyshev distance is applied. The higher the _t_ value, the smaller
the distance value measured. Considering a distance metric, the
coverage error of the representation can be stated as follows:

(
_R_ ( _N_ ) , _N_ ( _Z_ )
)
= max
_z_ ∈ _N_ ( _Z_ )
min
_z_ ′^ ∈ _R_ ( _N_ )
_d_ ( _z_ , _z_
′
) , (1)
which corresponds to the maximum distance from _z_ ∈ _N_ ( _Z_ ) to its
closest point in the representation _z_
′
∈ _R_ ( _N_ ). For the sake of sim-
plicity, from hereinafter, ( _R_ ( _N_ )) is used instead of( _R_ ( _N_ ) , _N_ ( _Z_ )).
A representation _R_ ( _N_ ) is said to have coverage γif ( _R_ ( _N_ ))  γ.
To increase coverage, the coverage error ( _R_ ( _N_ )) must be mini-
mized. Since the representation _R_ ( _N_ ) is a finite subset of _N_ ( _Z_ ) ,
minimizing the maximum distance from _z_ ∈ _N_ ( _Z_ ) to its closest
point in the representation _z_
′
∈ _R_ ( _N_ ) can be done, for bi-objective
problems, by minimizing the maximum distance between every
two consecutive points in the representation. Furthermore, by
guaranteeing that all consecutive points in the representation _R_ ( _N_ )
are distanced by a value lower or equal to γ, there is also the guar-
antee that _R_ ( _N_ ) has, at most, a coverage error of γ.
The uniformity level corresponds to the minimum distance be-
tween any two points in the representation:

(
_R_ ( _N_ )
)
= min
_z_ , _z_ ′^ ∈ _R_ ( _N_ ) , _z_  = _z_ ′^
_d_ ( _z_ , _z_
′
) (2)
A representation _R_ ( _N_ ) is said to have uniformity δif ( _R_ ( _N_ )) 
δ. To increase uniformity, ( _R_ ( _N_ )) should be maximized. As a con-
sequence, the discrete representation Problem 6 , can be stated as
a three-objective optimization problem ( Shao & Ehrgott, 2016 ):
min 
(
_R_ ( _N_ )
)
max 
(
_R_ ( _N_ )
)
min 
(
_R_ ( _N_ )
)
subject to: _R_ ( _N_ ) ⊂ _N_ ( _Z_ ) , 2  | _R_ ( _N_ ) | < ∞
(P6)
The objectives considered in Problem 6 are interrelated. Sayın
(20 0 0) noted that, by improving the representation coverage, car-
dinality would also increase. As uniformity increases, potentially,
the opposite effect on cardinality may emerge. Furthermore, Kidd
et al. (2020) proved that, for bi-objective problems, when compar-
ing two representations, _R_
′
( _N_ ) and _R_
′′
( _Z_ ) , with the same cardinal-
ity, where _R_
′′
( _Z_ ) is an equidistant representation, then ( _R_
′′
( _Z_ )) 
( _R_
′
( _N_ )) and( _R_
′′
( _Z_ ))  ( _R_
′
( _N_ )).
In this work, we make use of the Chebyshev norm to compute
( _R_ ( _N_ )) and ( _R_ ( _N_ )). Although other metrics may be used, the
choice is supported by three main aspects: (1) Chebyshev norm al-
lows the decoupling of the distance for each criterion, in the sense
that all coordinate distances will be at most the Chebyshev dis-
tance, i.e., if | _z_ (^) _k_ − _z_ ′^
_k_
|  1 for all _k_ = 1 ,... , _p_ , then _d_ (^) _l_ ∞ ( _z_ , _z_ ′^ )  1 ; (2)
From the previous fact, it also derives that both the coverage er-
ror and the uniformity level have a direct correspondence to the
coordinate distances without requiring the combination of multi-
ple coordinates; this allows greater control over the representation
criteria; and (3) The combination of coordinates, a fundamental
characteristic of the other metrics, may distort the distance met-
ric, and consequently the coverage error and uniformity level, if
the ranges of each criterion are significantly different ( Sayın, 20 0 0 ).
Using the Chebyshev norm implies that, in more than two objec-
tives, the minimization of the coverage error by minimizing the
maximum distance between each two consecutive points in the
representation can be decoupled for each criterion. Hence, by guar-
anteeing that in all criteria consecutive points are distanced by a
value lower or equal to γ, there is also the guarantee that the cov-
erage error of the _R_ ( _N_ ) is, at most, γ.

**3. Representation of the Pareto front**

This section presents the three Grid Point Based Algorithms

(GPBA) developed by us to generate a representation of the

Pareto front for MOILP Problem 1 : _GPBA-A_ , _GPBA-B_ and _GPBA-C_.

Each algorithm targets one dimension of the discrete representa-

tion Problem 6 (coverage, uniformity, and cardinality) with dif-

ferent strategies for exploring the feasible region. The first two

algorithms, _GPBA-A_ and _GPBA-B_ , are an extension of the ones

proposed by Eusébio et al. (2014) , aiming to improve coverage

and uniformity, respectively. The third algorithm, _GPBA-C_ , repre-

sents an improvement over Mavrotas and Florios (2013) algorithm,

and concerns mostly cardinality. However, all algorithms solving

Problem 5 proposed in this work share the same main procedure,

detailed in Fig. 1.

1. The first step of the algorithm consists of the computation of

the lower (pessimistic) and upper (optimistic) bounds of the

```
Pareto front, respectively z
nad
a and z
∗
```
. Although the optimistic

value is typically easily computed by maximizing each objective


```
Fig. 1. Flowchart for the generation algorithm.
```
function separately, the inverse cannot always be done to ob-

tain the pessimistic values. Many studies in the literature show

the difficulty of obtaining good estimations for those bounds

(see Alves & Costa, 2009; Isermann & Steuer, 1988 ). Conse-

quently, nadir point overestimation is common. All three pro-

posed methods can handle overestimated bounds, allowing any

strategy to compute the pessimistic values for each objective

function.

2. The objective function _z_ (^) _q_ to be directly considered in Problem
5 is then chosen, as well as the representation algorithm ( _GPBA-
A_ , _GPBA-B_ or _GPBA-C_ ). Depending on the representation algo-
rithm, the desired coverage error, uniformity level or the maxi-
mum cardinality of the representation must be chosen.

3. The third step consists of the initialization of the main loop by

enforcing the parameters  _k_ , in Problem 5 , to be the same as

the pessimistic values for each objective with a small pertur-

bation, resulting in Problem 5 being in its most relaxed form.

Additionally, the relative worst values for each objective func-

```
tion, z
w v
k
, are also initialized as the ideal values. z
w v
is a vector
```
that stores, for each objective function _z_ (^) _k_ , the worst value for
that objective function obtained on the current iteration over
the objective function’s loop. That value is used as the result
of that iteration, for objective function _z_ (^) _k_ , when adjusting the
corresponding parameter.

4. The fourth step consists of the comparison of the problem’s pa-

rameters against the results of the previously solved problem,


```
Fig. 2. Process to determine if the problem needs to be solved or can be skipped.
```
in order to check if a new criterion vector needs to be com-

puted or if it is going to yield a redundant result. That proce-

dure is shown in Fig. 2 :

(a) Select from the list of previous iterations, for each _k_ =

```
1 ,... , p , k  = q , a Problem P
′
such that the 
′
vector is equal
```
```
to or closer to the nadir point, i.e., z
nad
k
```
```
 
′
k
  k , for all
```
_k_ = 1 ,... , _p_ , _k_  = _q_.

(b) If such a problem does not exist or, if it exists, its solu-

tion does not lay in the bounds of Problem 5 when using

, then solve Problem 5 and return to the main procedure

(see Fig. 1 ).

```
(c) If such a Problem P
′
exists but it is marked as infeasible,
```
then return to the main procedure (see Fig. 1 ), considering

the problem as an infeasible one.

```
(d) If such a Problem P
′
exists and it is marked as feasible, and
```
its solution is within the bounds of Problem 5 , when us-

```
ing , i.e., z
′
k
```
  _k_ , for all _k_ = 1 ,... , _p_ , _k_  = _q_ , then return to

the main procedure (see Fig. 1 ) considering the solution of

```
Problem P
′
as the solution of Problem 5.
```
5. If the obtained (or selected from the set of previously computed

solutions) criterion vector is feasible, such a vector is saved

```
and the new z
w v
k
updated for all objective functions, with the
```
exception of the function _z_ (^) _q_ and the innermost loop _p_. Then,
the new  _p_ parameter is readjusted according to the algorithm
used. Otherwise, and in case the representation algorithm is not
_GPBA-A_ , i.e. not the coverage representation, the early exit loop
is used (see Fig. 3 ).

6. The process is repeated until the optimistic values is reached

in all loops, i.e., until Problem 5 reaches its most constrained

form.

The aforementioned acceleration strategies, the redundancy check-

ing ( Fig. 2 ) and the early exit from loop ( Fig. 3 ) procedures were

originally proposed by Zhang and Reimann (2014). While the lat-

ter is only based on Proposition 1 , the former resorts to both

```
Proposition 1 and Proposition 2. Let LP
 s
denote a linear relaxation
```
of Problem 5 :

```
Proposition 1. (Infeasibility) If Problem LP
 s
is infeasible, then
```
_Problem 5 is also infeasible._

```
Proposition 2. (Optimality) If x
∗
is the optimal solution of Problem
```
```
LP
 s
, and x
∗
is in the feasible region of Problem 5 , then x
∗
is also the
```
_optimal solution of Problem 5._

In the following subsections, we will detail the algorithms to

compute the parameters  _k_ , for Problem 5.

_3.1. Coverage grid point based representation (_ GPBA-A _)_

When looking for a representation that privileges the coverage

of the Pareto front, the algorithm will aim to represent all areas of

the Pareto front. This may be achieved, as discussed in Section 2.3 ,

by minimizing the maximum distance between any two consecu-

tive points in the representation ( ( _R_ ( _N_ )) ), Eq. (1) , i.e. the cov-

erage error. The quality of the representation is controlled by the

parameter γ, corresponding to the acceptable coverage error.

Parameter γ is sometimes difficult to determine. An alterna-

tive to directly conveying the acceptable coverage error is to de-

```
fine an acceptable cardinality in each objective ( π
k
, for all k =
```
```
1 ,... , p , k  = q ) based on their range ( z
∗
k
− z
nad
k
, for all k = 1 ,... , p ,
```
_k_  = _q_ ). The ratio between the range and cardinality provides the


```
Fig. 3. Accelerated exit algorithm.
```
acceptable coverage error for each objective: γ _k_ = ( _z_
∗
_k_
− _z
nad
k_
) /π _k_.

The sum of the specified cardinality in each objective will cor-

respond to the approximate cardinality of the representation, i.e.,

( _R_ ( _N_ )) =

∑ (^) _p
k_ = 1 , _k_  = _q_
π _k_. Note that the cardinality of the represen-
tation ( _R_ ( _N_ )) is merely an approximation since the distribution
of the Pareto front is unknown and it is not this parameter that
controls the representation.
Algorithm 2 presents the method to adjust parameter  _k_ when
aiming to have a representation that privileges coverage. It per-
forms a single run for every iteration and for each objective func-
tion _z_ (^) _k_ , i.e. for each loop, after solving a problem and obtaining,
or not, a new criterion vector. The algorithm takes six inputs: (1)
the desired coverage error over that loop, γ _k_ ; (2) the value of the
 _k_ parameter of Problem 5 used in the current iteration; (3) the
_k_ th component of the criterion vector _z_ , _z_ (^) _k_ , for the Problem 5 used
in the current iteration; (4) the _k_ th component of the ideal vector
_z_
∗
, _z_
∗
_k_
; (5) the _k_ th component of the approximation to the nadir
vector _z
nad_
, _z
nad
k_
; and (6) the set _D_ , corresponding to the ordered
set of the discarded points for the current loop, which contains
the points already obtained as well as the areas where redundant
points will be obtained. The algorithm’s outputs are the updated
value of parameter  _k_ to be used in the next iteration and the up-
dated ordered set of discarded points, _D_ , for the current loop. To
this end, the first part of the algorithm updates _D_ based on one of
the following two cases:
**Algorithm 2:** Adjusting the parameter  _k_ in _GPBA-A_.
**1 Input:** γ _k_ ,  _k_ , _z_ (^) _k_ , _z_ ∗
_k_
, _z nad_^
_k_
, _D_ ;
**2 Output:**  _k_ , _D_ ;
**3 if** ( _X_
 _s_
= {} ) **then
4** _D_ ← _D_ ∪ {
  _k_  ,  _k_  + 1 ,... , _z_
∗
_k_
} ;
**5 else
6** _D_ ← _D_ ∪ {
  _k_  ,  _k_  + 1 ,... , _z_ (^) _k_ } ;
**7 if** ( _k_ = _z
nad
k_
) **then
8**  _k_ ← _z_
∗
_k_
;
**9 else
10** Let _z_ 1 and _z_ 2 be the most distant consecutive values in _D_ ;
**11 if** ( _d_ ( _z_ 1 , _z_ 2 )  γ
_k_
) **then
12**  _k_ ← _z_
∗
_k_
+ 1 ;
**13** _D_ ← {} ;
**14 else
15**  _k_ ← ( _z_ 1 + _z_ 2 ) / 2 ;
**16 return** ( _k_ , _D_ ) ;

1. If Problem 5 solved in this iteration, when using  _k_ , is infea-

```
sible (the feasible region is empty, X
 s
= {} ), then if the prob-
```
lem is more constrained it will be infeasible too, as stated in

Proposition 1. As a result, all the points ranging from  _k_ to the

```
k th component of the ideal vector, z
∗
k
, can be discarded, and
```
should be added to _D_ for the current loop.

2. If Problem 5 solved in this iteration, when using  _k_ , is feasible,

all the points ranging between  _k_ to the _k_ th component of the

```
obtained vector, z
k
, can be discarded and should be added to D.
```
This is based on Proposition 2.

Afterwards, using the updated set of discarded points, _D_ , the

algorithm computes the new value for parameter  _k_ :

1. In the case where the current value of 
    _k_
       is the _k_ th compo-

```
nent of the pessimistic vector, z
nad
k
, the problem solved in this
```
iteration corresponds to the first extreme point. Hence, the fol-

lowing point to compute is the next extreme, the ideal value

```
z
∗
k
```
.

2. Otherwise, _D_ will have at least more than two points. As a re-

sult, the two consecutive points distancing the most from each

other are drawn from set _D_. Then, on the one hand, if the dis-

tance between those points meets the desired coverage error,

the loop is be exited. To do so, the set _D_ is updated as empty

and the  _k_ parameter is set to a value higher than the ideal

point value of objective function _z_ (^) _k_ , _z_
∗
_k_
+ 1. On the other hand,
if the distance between those points meets the desired cover-
age error,  _k_ is set to find a point that minimizes the coverage
error, i.e., a point at least in between the two most distant con-
secutive points, ( _z_ 1 + _z_ 2 ) / 2.
_3.2. Uniformity grid point based representation (_ GPBA-B _)_
A representation that privileges uniformity is a representation
which spreads points as uniformly as possible. By maximizing the
minimum distance between two points in the representation, the
uniformity level ( _R_ ( _N_ )) is increased. This representation is con-
trolled by an acceptable uniformity level δ.
Similar to the acceptable coverage level, the acceptable uni-
formity level may be difficult to define. It can be derived from
the range of each objective function values, ( _z_
∗
_k_
− _z
nad
k_
, for all _k_ =
1 ,... , _p_ , _k_  = _q_ ) and the acceptable cardinality in each objective ( π _k_ ,
for all _k_ = 1 ,... , _p_ , _k_  = _q_ ). The ratio between the range and cardi-
nality provides the acceptable uniformity level for each objective:
δ _k_ = ( _z_
∗
_k_
− _z
nad
k_
) /π _k_.
The procedure to determine  _k_ , for the next iteration, is pre-
sented in Algorithm 3. It takes as input the defined uniformity
level δ _k_ and the _k_ th component of the obtained vector, _z_ (^) _k_. The al-
gorithm in each iteration adds δ _k_ to _z_ (^) _k_ , until  _k_ is greater than the


**Algorithm 3:** Adjusting the parameter  _k_ in _GPBA-B_.

**1 Input:** δ _k_ ,  _k_ ;

**2 Output:**  _k_ ;

**3**  _k_ ← _z_ (^) _k_ + δ _k_ ;
**4 return** ( _k_ ) ;
maximum value for objective function _z_ (^) _k_ , _z_
∗
_k_

. In that case the loop

is exited because the problem becomes infeasible.

_3.3. Cardinality grid point based representation (_ GPBA-C _)_

The cardinality representation we propose provides a range in-

sensitive search strategy to distribute the desired number of cri-

terion vector in the feasible region. When the desired cardinality

is specified, it is very common that the ranges of the objectives

in which it is based are overestimated. As a result, the final car-

dinality of the representation will be significantly lower than the

originally considered, and in which the determination of parame-

ters, such as uniformity level δand coverage error γ, may have

been based. The cardinality representation algorithm addresses this

drawback by defining the search region in a way that intends to

maintain the originally determined cardinality, and provides a bal-

ance between uniformity and coverage.

The cardinality representation algorithm starts by defining a

uniform grid between the ideal and nadir points. Whenever a cri-

terion vector is computed that skips a step on the grid, the grid

is redefined from that point until the ideal value, considering only

the remaining points left to compute. This results in a refinement

of the search grid within the feasible region of the problem, and

consequently in a range insensitive search strategy.

Algorithm 4 presents the procedure to adapt parameter  _k_. It

takes six inputs: (1) the grid starting point for that loop, _z
start
k_

; (2)

the _k_ th component of the ideal vector _z_
∗
, i.e. the grid end point

for that loop, _z_
∗
_k_
; (3) the number of points to obtain from that grid,

π
′
_k_
; (4) the position within the grid for this objective, i.e. the grid

point number, _i_ (^) _k_ ; (5) the _k_ th component of the criterion vector _z_ ,
_z_ (^) _k_ , for the Problem 5 used in the current iteration; and (6) the _k_ th
slack variable, _s_ (^) _k_ , for the Problem 5. For the first iteration, prior to
entering Algorithm 4 , the grid start point, _z start_^
_k_
, corresponds to the
nadir approximation vector, _z
nad
k_
, the position within the grid, _i_ (^) _k_ ,
to zero, and the number of points to obtain, π
′
_k_
, to the cardinality
for that objective minus 1, π _k_ − 1. The algorithm outputs the new
parameter  _k_ and the updated grid parameters, _z
start
k_
, π
′
_k_
and _i_ (^) _k_. To
that end, the following procedure is applied:

1. The first step determines if the criterion vector from the prob-

lem solved in the previous iteration, _z_ (^) _k_ , makes any of the next
grid points redundant according to Proposition 2 , i.e. if a step
on the grid may be skipped. Accordingly, the grid step size,
_step_ , is computed by dividing the grid range by the number of
**Algorithm 4:** Adjusting the parameter  _k_ in _GPBA-C_.
**1 Input:** _z
start
k_
, _z_
∗
_k_
, π
′
_k_
, _i_ (^) _k_ , _z_ (^) _k_ , _s_ (^) _k_ ;
**2 Output:**  _k_ , _i_ (^) _k_ , π
′
_k_
, _z
start
k_
;
**3** _step_ ← max
{
( _z_
∗
_k_
− _z
start
k_
) /π
′
_k_
, 1
}
;
**4** _b_ ← | _s_ (^) _k_ / _step_ | ;
**5 if** ( _b_ > 0 ) **then
6** _z start_^
_k_
← _z
k_
;
**7** π
′
_k_
← π
′
_k_
− _i_ (^) _k_ ;
**8** _i_ (^) _k_ ← 1 ;
**9** _step_ ← max
{
( _z_
∗
_k_
− _z
start
k_
) /π
′
_k_
, 1
}
;
**10 else
11** _i_ (^) _k_ ← _i_ (^) _k_ + 1 ;
**12**  _k_ ← _z
start
k_
+ _i_ (^) _k_ × _step_ ;
**13 if** ( _k_ > _z_
∗
_k_
) **then
14** _z
start
k_
← _z
nad
k_
;
**15** π
′
_k_
← π _k_ ;
**16** _i_ (^) _k_ ← 0 ;
**17 return** ( _k_ , _z
start
k_
, _i_ (^) _k_ , π
′
_k_
) ;
grid points in that grid. The integer part of the division between
the slack variable, _s_ (^) _k_ , and the grid step size, _step_ , determines
how many grid points may be skipped.
(a) In case there are steps on the grid to skip, the grid is re-
defined. The new starting point for the grid, _z
start
k_
, will be-
come the corresponding component of the criterion vector
obtained in the current iteration, _z_ (^) _k_. The updated number of
grid points for the new grid, π
′
_k_
, corresponds to the num-
ber of points from the current grid subtracted the number
of grid points already computed, _i_ (^) _k_. The updated position
within the grid, _i_ (^) _k_ , will be the first, at the beginning of the
new grid. Finally, the step according to the newly defined
grid is computed.
(b) Otherwise, the grid is not updated and the position within
the grid, _i_ (^) _k_ , is incremented.

2. The second step makes use of the grid parameters computed in

```
the previous step, z
start
k
```
, _i_ (^) _k_ and _step_ , to update parameter  _k_. In
case  _k_ surpasses the ideal vector value _z_
∗
_k_
, the loop is exited
and the grid parameters return to their initial values: the grid
start point, _z start_^
_k_
, corresponds to the nadir approximation vec-
tor, _z
nad
k_
; the number of points to obtain, π
′
_k_
, to the cardinality
for that objective, π _k_ ; and the position within the grid, _i_ (^) _k_ , to
zero.
_3.4. An illustrative example_
Consider the following numerical example, originally presented
in Isermann and Steuer (1988) :
max _z_ 1 ( _x_ ) = 2 _x_ 1 − 2 _x_ 4 − 2 _x_ 6 − 2 _x_ (^7)
max _z_ 2 ( _x_ ) = − 2 _x_ 1 + _x_ 2 + 2 _x_ 3 − _x_ 4 + _x_ 5 + 2 _x_ 6 − _x_ (^7)
max _z_ 3 ( _x_ ) = − _x_ 1 − 2 _x_ 2 − 2 _x_ 4 + 3 _x_ 5 + _x_ (^6)
subject to: _x_ 1 + _x_ 2 + 3 _x_ 3 + 3 _x_ 5 + 2 _x_ 6  61
3 _x_ 2 + 2 _x_ 3 + 4 _x_ 4  72
5 _x_ 1 + 3 _x_ 2 + 5 _x_ 5 + 4 _x_ 6 + 4 _x_ 7  76
4 _x_ 1 + 2 _x_ 2 + 4 _x_ 4 + 4 _x_ 6  51
5 _x_ 1 + 2 _x_ 2 + 3 _x_ 4 + _x_ 5 + 4 _x_ 6  66
2 _x_ 1 + 2 _x_ 2 + 4 _x_ 4 + 4 _x_ 5 + 4 _x_ 6 + 5 _x_ 7  59
3 _x_ 1 + 2 _x_ 3 + 5 _x_ 5 + _x_ 6 + 2 _x_ 7  77
_x_ 1 , _x_ 2 , _x_ 3 , _x_ 4 , _x_ 5 , _x_ 6 , _x_ 7 ∈ Z
+
0
.
This section illustrates the three representation algorithms,
_GPBA-A_ , _GPBA-B_ , and _GPBA-C_. The first iteration over the outermost


loop is used and the results represented in Appendix A. Common

to all representations is the need to compute both the ideal point

and the nadir point, or, alternatively, an approximation of the nadir

point. In this case, the ideal point is _z_
∗
= ( 24 , 49 , 42 ) , and the nadir

approximation point, obtained through the minimization of each

individual objective function, is _z
nad_
= (− 28 , − 28 , − 48 ). Addition-

ally, _z_ 1 will be used as _z_ (^) _q_ , _z_ 2 as the outermost loop, and _z_ 3 as
the innermost loop. Hence, all algorithms start with  2 = − 28 and
 3 = − 48 , as well as with the relative worst value for the outer-
most loop _z
w v_
2
= 49.
_3.4.1. Coverage grid point based representation (_ GPBA-A _)_
Assume the acceptable coverage error for the representation be-
ing computed is γ= 15 , for all objective functions. The first itera-
tion over the outermost loop will have  2 = − 28. Then, the itera-
tions over the innermost loop, _k_ = 3 , are the following:

1. The first iteration over the innermost loop ( Fig. A.4 a) solves

```
Problem 5 using  2 = − 28 and  3 = − 48. Point z
1
= ( 24 , 9 , − 14 )
```
```
is obtained and added to the representation R ( N ) = { z
1
}. Next,
```
```
the relative worst value for objective function z 2 , z
w v
2
, and
```
```
the ordered set of discarded points, D , are updated to z
w v
2
= 9
```
and _D_ = {− 48 , − 47 ,... , − 14 }. Since the current value for  3 is

```
z
nad
3
, the new value for  3 is 42, corresponding to the ideal
```
value.

2. Applying the flowchart on Fig. 2 , Problem 5 needs to be solved

for  2 = − 28 and  3 = 42 , and the point obtained is _z_ 2 =

```
( 0 , 20 , 42 ) ( Fig. A.4 b). Point z
2
is added to the representation,
```
```
resulting in R ( N ) = { z
1
, z
2
}. Since z
2
2
 z
w v
2
, the worst value for
```
```
objective function z 2 does not need to be updated. z
2
3
is added
```
to the set of discarded points _D_ = {− 48 , − 47 ,... , − 14 , 42 }. The

two most distant consecutive values in _D_ are − 14 and 42, hence

the new value of  3 is ( 42 + (− 14 )) / 2 = 14.

3. Again, applying the flowchart on Fig. 2 , Problem 5 is solved

```
for  2 = − 28 and  3 = 14 , obtaining z
3
= ( 14 , 13 , 14 ) ( Fig. A.4 c),
```
```
and updating the representation to R ( N ) = { z
1
, z
3
, z
2
}. Again,
```
```
the worst value for objective function z 2 , z
w v
2
, does not re-
```
quire updating, while the set of discarded points is updated

to _D_ = {− 48 , − 47 ,... , − 14 , 14 , 42 }. At this point the maximum

distance between points _D_ is 28. As a result, the new value for

 3 is ( 14 + (− 14 )) / 2 = 0.

4. Problem 5 is solved for  2 = − 28 and  3 = 0 , obtaining _z_
    4
       =

( 22 , 6 , 1 ) ( Fig. A.4 d), and updating the representation to _R_ ( _N_ ) =

```
{ z
1
, z
4
, z
3
, z
2
}. The worst value for objective function z 2 , z
w v
2
,
```
```
needs to be updated z
w v
2
= 6. The set of discarded points is up-
```
dated to _D_ = {− 48 , − 47 ,... , − 14 , 0 , 1 , 14 , 42 }. The new value for

 3 is ( 42 + 14 ) / 2 = 28.

5. Using  2 = − 28 and  3 = 28 , Problem 5 is solved and _z_
    5
       =

```
( 8 , 13 , 29 ) is obtained ( Fig. A.4 e). criterion vector z
5
is then
```
```
added to the representation, R ( N ) = { z
1
, z
4
, z
3
, z
5
, z
2
}. The
```
worst value for objective function _z_ 2 remains the same,

```
z
w v
2
= 6. The set of discarded points is updated to D =
```
{− 48 , − 47 ,... , − 14 , 0 , 1 , 14 , 28 , 29 , 42 }. At this point, the maxi-

mum distance between points in the set of discarded points is

14, which is less than γ. Hence, the loop stops, and the cov-

erage error obtained for the first iteration over the outermost

loop was 15.

For the next iteration over the loop of the objective function _z_ 2 ,

the value of _z
w v_
2
= 6 is used as _z_ 2 for the discarded vector of that

loop.

_3.4.2. Uniformity grid point based representation (_ GPBA-B _)_

Suppose the objective is to have a representation with at least

a uniformity level of δ= 10 over all objective functions. The algo-

rithm begins with  2 = − 28 , for the outermost loop, and the fol-

lowing iterations occur over the innermost loop:

1. In the first iteration ( Fig. A.5 a) Problem 5 is solved using  3 =

```
− 48. Point z
1
= ( 24 , 9 , − 14 ) is obtained and added to the repre-
```
```
sentation R ( N ) = { z
1
}. The new  3 is updated according to the
```
```
desired uniformity level as:  3 = z
1
3
+ δ 3 = − 14 + 10 = − 4. Be-
```
fore the next iteration the worst value for objective function _z_ 2 ,

```
z
w v
2
, is also updated to 9.
```
2. The second iteration ( Fig. A.5 b) uses  3 = − 4 to solve Problem

5 and point _z_ 2 = ( 24 , 5 , − 3 ) is obtained and added to the rep-

```
resentation, R ( N ) = { z
1
, z
2
}. The third component of the vec-
```
```
tor is updated accordingly, becoming  3 = 7. Since z 2
2
< z w^ v^
2
, the
```
worst value for objective function _z_ 2 is updated, resulting in

```
z
w v
2
= 5.
```
3. When solving Problem 5 with  2 = − 28 and  3 = 7 ( Fig. A.5 c)

```
point z
3
= ( 18 , 8 , 9 ) is obtained and added to the represen-
```
```
tation, R ( N ) = { z
1
, z
2
, z
3
}. The  3 is updated as  3 = z
3
3
+ δ 3 =
```
```
9 + 10 = 19. Again, since z
2
2
 z
w v
2
, the worst value for objective
```
function _z_ 2 is not updated.

4. In the fourth iteration ( Fig. A.5 d) Problem 5 is solved using

```
 3 = 19. The criterion vector is z
4
= ( 12 , 11 , 21 ) , which is added
```
```
to the representation, R ( N ) = { z
1
, z
2
, z
3
, z
4
}. As a consequence,
```
```
 3 is updated as z 4
3
+ δ 3 = 21 + 10 = 31. In this iteration, be-
```
```
cause z
2
2
 z
w v
2
, the worst value for objective function z 2 does
```
```
not need to be updated, remaining at z w^ v^
2
= 5.
```
5. For the fifth iteration ( Fig. A.5 e) Problem 5 is solved using  3 =

```
31. The resulting criterion vector is z
5
= ( 6 , 14 , 33 ) , which is
```
```
added to the representation, becoming R ( N ) = { z
1
, z
2
, z
3
, z
4
, z
5
}.
```
```
Once more, z
w v
2
is not updated since z
w v
2
< 14. Then,  3 is up-
```
dated following Algorithm 3. However, since the resulting  3 is

```
43, which is larger than z
∗
3
= 42 , the loop is finished (see Fig. 1 ).
```
For the next iteration over the outermost loop, objective func-

```
tion z 2 , the value of z w^ v^
2
is used as z 2 to compute the next value of
```
 2 ,  2 = _z_ 2 + δ 2 = 5 + 10 = 15.

_3.4.3. Cardinality grid point based representation (_ GPBA-C _)_

Consider a desired cardinality for the representation of 25

points, corresponding to a division of each constrained objective

function _z_ (^) _k_ , _k_ = 1 ,... , _p_ , _k_  = _q_ , into 5. The first iteration over the
outermost loop, _k_ = 2 , starts with  2 = _z
nad_
2
= − 28 and _z
w v_
2
= _z_
∗
2
=
49 , and then iterates over the innermost loop in the following way:

1. For the first iteration, in Fig. A.6 a,  3 is − 48 , corresponding to

```
the nadir approximation value for that objective, z nad^
3
```
. Solving

Problem 5 with those parameters, yields the criterion vector

```
z
1
= ( 24 , 9 , − 14 ) , which is added to the representation R ( N ) ,
```
```
and we update z
w v
2
with 9. Since this is the first iteration,
```
```
the grid start point z
start
3
is − 48 , i.e., the nadir approximation
```
value, the position within the grid, _i_ 3 , is 0, and the number

```
of points to obtain with such a grid, c
′
3
, is equal to the objec-
```
tive’s cardinality minus one, 4. The slack variable for Problem

5 , _s_ 3 , is the difference between  3 and the third component of

the criterion vector, i.e., _s_ 3 = − 14 −(− 48 ) = 34. To determine

whether a step in the grid may be skipped or not, we first

compute the step size: _step_ = max

```
{
( 42 −(− 48 )) / 4 , 1
```
```
}
= 22. 5.
```
Since the number of steps to skip, _b_ , is the integer part of

34 / 22. 5 = 1. 51 , then one step in the grid must be skipped. Con-

```
sequently, the grid is refined from the obtained points, z
start
3
=
```
```
z
1
3
= − 14 , onwards, considering the number of points left to
```
```
compute: c
′
3
= 4 − 0 = 4 , step = max
```
```
{
( 42 −(− 14 )) / 4 , 1
```
```
}
= 14 ,
```
_i_ 3 = 1 ( Fig. A.6 b). Hence, the 3 parameter for the following it-

eration is  3 = − 14 + 1 × 14 = 0.

2. For the second iteration, in Fig. A.6 b, we use  3 = 0 when solv-

```
ing Problem 5 , leading to the vector z
2
= ( 22 , 6 , 1 ) , which is
```
added to the representation, _R_ ( _N_ ) = { _z_ 1 , _z_ 2 }. The worst value


for objective function _z_ 2 needs to be updated with 6, since

```
z 2
2
< z w^ v^
2
```
. As a result, the third slack variable for Problem 5 is

_s_ 3 = 1 − 0 = 1. Since the step size is 14, no point is skipped,

and it can be determined by the integer part of _b_ =  1 / 14  = 0.

Hence the grid remains the same and the position within the

grid is increased by one, _i_ 3 = 2. Hence, the  3 parameter for the

following iteration is  3 = − 14 + 2 × 14 = 14.

3. For the third iteration, in Fig. A.6 c, we solve Problem 5 with

```
 3 = 14 , leading to vector z
3
= ( 14 , 13 , 14 ) , which is added to
```
```
the representation, R ( N ) = { z
1
, z
2
, z
3
}. The worst value for ob-
```
```
jective function z 2 , z
w v
2
, remains the same. The slack variable
```
for Problem 5 is _s_ 3 = 0 , hence no point is skipped and the grid

is kept. The position within the grid is increased by one, _i_ 3 = 3 ,

and  3 parameter is updated to  3 = − 14 + 3 × 14 = 28.

4. For the forth iteration, in Fig. A.6 d, we use  3 = 28 , which re-

```
sults in the criterion vector z
4
= ( 8 , 13 , 29 ) , and the represen-
```
```
tation R ( N ) = { z
1
, z
2
, z
3
, z
4
}. The worst value for objective func-
```
```
tion z 2 keeps its value z
w v
2
= 6 , because z
4
2
 z
w v
2
```
. Since _z_
    4
    3
       − 3 =

1 corresponds to the slack variable, there are no redundant

points in the grid, _b_ =  1 / 14  = 0. As a result the grid remains

the same and the position within the grid is increased by one,

_i_ 3 = 4. The  3 parameter is then updated to  3 = − 14 + 4 × 14 =

42.

5. For the fifth and last iteration in this loop, in Fig. A.6 e, we use

```
 3 = 42 , which results in the criterion vector z
5
= ( 0 , 20 , 42 ) ,
```
updating the representation to _R_ ( _N_ ) = { _z_ 1 , _z_ 2 , _z_ 3 , _z_ 4 , _z_ 5 }. Since

```
 3 = z
5
3
no point is skipped and the position within the grid
```
may be increased by one, _i_ 3 = 5. Due to _i_ 3 being greater

```
than c
′
3
, the updated value of  3 = − 14 + 5 × 14 = 56 is greater
```
```
than the ideal value for objective function z 3 , z
∗
3
= 42. Hence,
```
the loop will be exited and the grid control parameters for

this objective, _k_ = 3 , will return to the values of the first

iteration.

For the next iteration over the outermost loop, objective func-

tion _z_ 2 , the value of _z
w v_
2
= 6 is used as _z_ 2 to compute the

next value of  2 , using Algorithm 4. Because the slack vari-

able _s_ 2 = − 28 − 6 = − 34 and the step size to _step_ = max

```
{
( 49 −
```
(− 28 )) / 4 , 1

```
}
= 19. 25 , one step is skipped, b = | − 34 / 19. 25 | =
```
1. The grid parameters are adapted to _z
start_
2
= _z_ 2 = 6 , _c_
′
2
= 4 −

0 = 4 and _i_ 2 = 1. The step size is updated to _step_ = max

```
{
( 49 −
```
6 ) / 4 , 1

```
}
= 10. 75 and, at last, 2 = 6 + 1 × 10. 75 = 16. 75.
```
_3.5. Generating the entire Pareto front_

As the three algorithms presented in the above section, _GPBA-_

_A_ , _GPBA-B_ , and _GPBA-C_ , are based on solving a sequence of varia-

tions of Problem 5 , they can be used to generate the whole Pareto

front under the condition that the objective function only takes in-

teger values. Additionally the parameters that control each algo-

rithm, the acceptable coverage error, the uniformity level, and the

cardinality, must be chosen to ensure the production of the entire

Pareto set.

When using algorithm _GPBA-A_ to compute the entire Pareto

front, the desired acceptable coverage error, γ, is 1, guaranteeing

that the distance between any two consecutive points in the Pareto

front representation is at most unitary. Since these algorithms are

applied to problems with integer objective functions, if  1 , then

is, in fact, equal to 1.

In case algorithm _GPBA-B_ is used, the acceptable uniform level,

δ, should be set to 1 to achieve the computation of the entire

Pareto front. Setting δ= 1 ensures the algorithm will search for the

next solution as close to the previous as a unitary step in the ob-

jective space.

Finally, for the case of _GPBA-C_ , when generating the entire

Pareto front, the cardinality should be set for each objective as the

respective range. As a result, a unitary step size is used for each

objective function’s loop and the algorithm restricts the search re-

gion for the next solution to a unitary distance in the objective

space.

The three proposed algorithms were tested on multiple in-

stances on their performance when generating the whole Pareto

front. To that end, the above parameters were used. Results of

those experiments are presented in Section 4.

**4. Computational experiments**

This section presents the design of the experiments, some im-

plementation issues, the computational results for the identifica-

tion of the whole Pareto front, some experiments on the represen-

tations of the Pareto front, and a few final comments and remarks.

The performance of the three proposed algorithms _GPBA-A_ ,

_GPBA-B_ and _GPBA-C_ is compared with the works of Mavrotas and

Florios (2013) , Zhang and Reimann (2014) and Nikas et al. (2020) ,

hereinafter referred to as _AUGM-2_ , _S-AUGM_ and _R-AUGM_ , respec-

tively. All algorithms, apart from _R-AUGM_ were coded using Py-

Charm Community Edition 2020 for Windows Desktop, Python 3.

as the interpreter and CPLEX 12.9 as the optimization solver. For

_R-AUGM_ , the code provided by the authors in https://github.com/

KatforEpu/Augmecon-R was adapted regarding the instances. All

experiments were performed in a workstation equipped with two

processors Intel Xeon X5680 3.33GHz and 24GB of RAM, running

Windows 10.

The code for the algorithms proposed in Section 3 as well

as the problem instances used in this section are available in

the following website: https://fenix.tecnico.ulisboa.pt/homepage/

ist175325/apresentacao.

_4.1. The design of the experiments_

The generation of the instances used in these experiments was

performed through a generator developed by the second author

```
of this study
1
```
. This generator makes use of the random number

generation procedure of NETGEN generator proposed in Klingman,

Napier, and Stutz (1974) and requires as input the following ele-

ments:

- The number of variables ( _n_ ), objective functions ( _p_ ) and con-

straints ( _m_ ).

- The ranges for the parameter values (coefficients of the con-

straints and the weights/profits of the objective functions).

These parameters are randomly generated by using a uni-

form distribution with a specific seed number for each con-

straint/objective function.

- In this study, all seed numbers are increased by a fixed value,

for each instance generated.

- The right-hand side of each constraint is bounded by half the

sum of the constraint’s coefficients.

The algorithm’s performance is essentially compared in terms

of solution quality and running time, or a proxy of the latter, such

as the number of iterations per non-dominated criterion vector

computed. As such, since both running time and its proxies are of-

ten not normally distributed, each problem type was tested on 30

instances. The choice of 30 instances was motivated by the con-

clusion of Coffin and Saltzman (20 0 0) that, for those measures

of performance, 30 to 40 observations would guarantee statisti-

cally significant results. Apart from the aforementioned observa-

tion, Coffin and Saltzman (20 0 0) also provided other suggestions

to draw meaningful conclusions from the comparison of algorithms

(^1) For more details please contact José Rui Figueira at figueira@tecnico.ulisboa.pt


that were followed in this work, such as the use of sets of prob-

lems where certain parameters are varied, and the use of mean

and variance of metrics for comparison. As a result, a total of 570

different uncorrelated instances were generated.

_4.2. Implementation issues_

All algorithms, aside from _AUGM-2_ , can overcome poor nadir

estimations, since the constraints’ right-hand-side is computed

based on the obtained solution and not beforehand. Hence, for

those algorithms, in these experiments, we simply use the sin-

gle minimization of each objective function individually as a nadir

point. However, since the improvements on the constrained ob-

jective functions are weighted by the width of the objective’s

range, due to some solvers’ sensitivity, the value of parameter ρ

in Problem 5 may have to be adapted for some instances. That

is, when the slack variable value is low, and it is divided by the

width of the objective’s range and multiplied by a small num-

ber ρ, this product may become smaller than the software’s sen-

sitivity. Hence, one way to go around this implementation issue

is to increase the value of ρ. The consequence of not adapting

parameter ρis the failure to compute some non-dominated cri-

terion vectors. For the results presented in this paper, ρ= 10
− 3

was used for algorithm _AUGM-2_ and ρ= 10
− 2
for the remaining

algorithms.

Algorithm _AUGM-2_ , however, requires good quality nadir es-

timations. For such purpose, as suggested in Mavrotas and Flo-

rios (2013) , the lexicographic pay-off table is used to provide an

approximation of the nadir point. However, as it has been dis-

cussed in literature, lexicographic pay-off tables do not guarantee a

good estimation of nadir points for more than two objective func-

tions (see, e.g., Alves & Clímaco, 2007; Isermann & Steuer, 1988 ).

For this implementation reason, in some instances, _AUGM-2_ would

not explore the whole Pareto front and, consequently, skip non-

dominated criterion vectors. To prevent this from happening, as

proposed by Mavrotas and Florios (2013) , the objective functions’

range provided by the lexicographic payoff table were expanded

by 15%.

Algorithm _GPBA-A_ requires recording the set of discarded

points, _D_. There are as many _D_ sets as the number of constrained

objectives, i.e. _p_ − 1 , and the set is temporary only existing while

still in the same computation over the previous loop. The size of

each set _D_ can be at most the size of the integer size of the ob-

jective functions’ ranges. This can create memory overflow issues.

However, in the experiments described in the remaining of this pa-

per, that problem never appeared. Nevertheless, if that issue arises,

a better nadir point estimation than the single minimization of

each objective function individually is required. In such case, the

range provided by the lexicographic payoff table can be expanded

by 10%/15%.

The implementation of _R-AUGM_ provided by the authors of

Nikas et al. (2020) requires the pre-computation of the objec-

tive functions’ range to determine the number of grid points.

That procedure was done using as nadir estimation the single

minimization of each objective function individually as a nadir

point.

_4.3. Experiments on generating the whole Pareto front_

To evaluate the computational and accuracy behaviour of dif-

ferent algorithms, when computing the Pareto front, experiments

were performed on binary and integer instances, considering both

single and multi-dimensional cases. Additionally, 180 tests were

run on multi-objective general integer programming instances. This

was done in order to, not only compare the algorithms, but also

understand how changes in the number of objective functions,

constraints, variables, and problem types affect the performance of

the algorithms.

In the remaining of this section, analysis of the results for the

aforementioned experiments are presented. This analysis is accom-

panied by summary tables, where underlined results show algo-

rithms’ drawbacks; and results in bold show algorithms’ strengths.

Algorithms are evaluated on the average and standard deviation

over the number of unique non-dominated criterion vectors found

( | _N_ ( _Z_ ) | and σ _N_ ( _Z_ ) )and CPU time, as well as the average number

of iterations computed for each non-dominated criterion vectors

found ( _iter_ / _z_ ˆ^ ). Although the latter corresponds to a proxy of CPU

time, it is included in this study because it is not correlated to the

number of criterion vectors in the Pareto front neither with the

implementation software while CPU time typically is. It therefore

coveys better the influence of the remaining problem parameters

on the algorithm’s performance.

_4.3.1. Multi-objective_ { 0 , 1 }− _knapsack instances_

The first set of problems studied were multi-objective binary

knapsack instances with three and four objective functions, con-

sidering one constraint. The three-objective instances have 50, 75,

and 100 variables, while the four-objective instances were gener-

ated with just 50 variables. Both the objective function’s profits

and the constraint’s coefficients are drawn from a uniform distri-

bution of type _U_ [1 , 100]. The initial seed numbers for the three-

objective instances with 50 variables were 128, 888 and 6 for each

objective function profit, and 40 for the constraint’s coefficients.

For the remaining three-objective instances, the seed numbers for

the objective functions’ profits were 47, 28 and 626, respectively,

whilst the seed number 135 was used for the constraint’s coeffi-

cients. Lastly, for the four-objective instances, the seed numbers of

the objective functions’ profits are 47, 28, 626 and 135, and 298

for the constraint’s coefficients. All seed numbers are increased by

5 on each instance of each type. For each set of problems, 30 in-

stances were generated. Table 1 presents experiments results for

the above described instances.

Regarding the instances with three objectives, the computa-

tional time increases significantly with the number of criterion

vectors in the Pareto front. However, the increase in the number

of variables does not impact the number of iterations per non-

dominated criterion vectors. All algorithms computed the same

number of non-dominated criterion vectors. Algorithm _AUGM-_

is the least efficient method among the ones studied, both in

terms of computational time and in terms of number of iter-

ations per non-dominated criterion vector obtained. The reason

for this is that, in this algorithm, iteration skipping and early

exit mechanisms are only applied to the innermost loop. Further-

more, as explained in Section 4.2 , _AUGM-2_ has a smaller space

to search since it is given a better nadir approximation than the

other algorithms. _GPBA-A_ also does not present competitive re-

sults since more iterations are performed per non-dominated cri-

terion vector computed for this algorithm when compared to the

others, which is also reflected in the computational time. _GPBA-A_

presents this behaviour due to the fact that the accelerated early

exit strategy cannot be applied to this algorithm. The results dis-

played by algorithms _AUGM-2_ and _GPBA-A_ justify the need and

added value of the integration of the acceleration strategies ( Figs. 2

and 3 ). _R-AUGM_ , although presenting better results than _GPBA-_

_A_ , performs worse than the other proposed algorithms in terms

of number of iterations done per non-dominated criterion vector.

Consequently, algorithms _AUGM-2_ , _GPBA-A_ and _R-AUGM_ were dis-

carded from further analysis. Finally, the performances of _S-AUGM_

and the other two proposed algorithms, _GPBA-B_ and _GPBA-C_ , are

similar.

In the case of the four-objective instances, for the aforemen-

tioned reasons, only algorithms _GPBA-B_ and _GPBA-C_ are compared


**Table 1**

Multi-objective 0–1 knapsack instances.

_p n_ Algorithm | _N_ ( _Z_ ) | σ _N_ ( _Z_ ) _iter_ / ˆ _z CPU_ (sec) σ _CPU_

```
3 50 AUGM-2 408.27 210.30 67.02 1083.47 599.
R-AUGM 408.27 210.30 1.99 183.02 77.
S-AUGM 408.27 210.30 1.89 58.36 34.
GPBA-A 408.27 210.30 14.37 394.56 238.
GPBA-B 408.27 210.30 1.89 58.36 34.
GPBA-C 408.27 210.30 1.89 58.61 34.
```
```
75 AUGM-2 1518.83 920.27 49.18 3523.38 1469.
R-AUGM 1518.83 920.27 1.89 961.57 449.
S-AUGM 1518.83 920.27 1.82 263.34 193.
GPBA-A 1518.83 920.27 11.69 1771.25 1378.
GPBA-B 1518.83 920.27 1.83 260.03 190.
GPBA-C 1518.83 920.27 1.83 260.18 189.
```
```
100 AUGM-2 3356.40 1628.35 42.07 9806.73 3775.
R-AUGM 3356.40 1628.35 1.84 2524.40 1011.
S-AUGM 3356.40 1628.35 1.79 666.86 411.
GPBA-A 3356.40 1628.35 9.28 4501.94 2674.
GPBA-B 3356.40 1628.35 1.79 661.26 427.
GPBA-C 3356.40 1628.35 1.79 662.42 427.
```
```
4 50 S-AUGM 2655.43 1392.73 4.74 9940.89 7008.
GPBA-B 2655.43 1392.73 4.74 10104.69 7152.
GPBA-C 2655.43 1392.73 4.77 10032.92 7205.
```
**Table 2**

Multi-objective multi-dimensional 0–1 knapsack instances.

_p n m_ Algorithm | _N_ ( _Z_ ) | σ _N_ ( _Z_ ) _iter_ / _z_ ˆ _CPU_ (sec) σ _CPU_

```
3 25 2 S-AUGM 80.03 38.39 1.91 8.06 4.
GBPA-B 80.03 38.39 1.91 8.11 4.
GBPA-C 80.03 38.39 1.91 8.30 4.
```
```
3 S-AUGM 84.83 40.56 1.92 8.95 4.
GPBA-B 84.83 40.56 1.92 8.99 4.
GPBA-C 84.83 40.56 1.92 9.03 4.
```
```
4 S-AUGM 88.30 44.96 1.92 9.67 5.
GPBA-B 88.30 44.96 1.92 9.78 5.
GPBA-C 88.30 44.96 1.92 9.73 5.
```
```
5 S-AUGM 90.73 37.97 1.92 9.94 4.
GPBA-B 90.73 37.97 1.92 10.02 4.
GPBA-C 90.73 37.97 1.92 10.12 4.
```
with _S-AUGM_. Analysis of the results shows for the three compared

algorithms high sensitivity of the number of required iterations to

the increase in the number of objectives. Furthermore, although

all algorithms perform similarly, _GPBA-C_ appears to have the low-

est efficiency having performed, on average, the most iterations

per non-dominated criterion vector found. This could be explained

by possibly needing a readjustment of parameter ρin Problem 5 ,

as explained in Section 4.2. However, when comparing in terms

of computational time, _GPBA-C_ is slightly more efficient than

_GPBA-B_.

_4.3.2. Multi-objective multi-dimensional_ { 0 , 1 }− _knapsack instances_

For the multi-objective multi-dimensional binary knapsack

problems, instances were generated varying the number of con-

straints between 2 and 5, with a fixed number of objectives at 3

and variables at 25. Both the objective function’s profits and the

constraint’s coefficients follow a uniform distribution bounded be-

tween 1 and 100. The seed numbers used for the objective func-

tion’s profits were 47, 63 and 728, while for each constraint equa-

tion’s coefficient the values used were 626, 135, 28, 17, 758, re-

spectively. Table 2 shows the results for algorithms _S-AUGM_ , _GPBA-_

_B_ and _GPBA-C_. For the reasons described in Section 4.3.1 , related to

poor performance, algorithms _AUGM-2_ and _GPBA-A_ are not consid-

ered in this comparison.

Table 2 shows that the performance of the algorithms is not

affected by the increase in the number of constraints. Addi-

tionally, the three tested algorithms, _S-AUGM_ , _GPBA-B_ and _GPBA-_

_C_ , are comparable in performance, finding the whole Pareto

front in a short amount of time. _S-AUGM_ appears to be slightly

more efficient when comparing average computational time, how-

ever differences between algorithms represent, in the worst

case, 1. 8%.

_4.3.3. Multi-objective integer knapsack instances_

To test the algorithms on integer multi-objective knapsack in-

stances, the instances generated in Section 4.3.1 with three ob-

jectives and 50 variables were solved using integer variables. In-

stance number 27, that uses seed numbers 158, 1018 and 136 for

the objective function’s profits and 170 for the constraint’s co-

efficients, was discarded, having exceeded a ten-hour computa-

tional time limit, due to having many non-dominated criterion

vectors. Hence, the results presented in Table 3 only consider

29 instances.

Although all algorithms are able to compute the whole Pareto

front, their performance differs. Whilst _S-AUGM_ takes less com-


**Table 3**

Multi-objective integer knapsack instances.

_p n_ Algorithm | _N_ ( _Z_ ) | σ _N_ ( _Z_ ) _iter_ / _z_ ˆ _CPU_ (sec) σ _CPU_

```
3 50 S-AUGM 3152.59 9061.66 1.98 2590.92 6666.
GPBA-B 3152.59 9061.66 1.71 2662.00 7017.
GPBA-C 3152.59 9061.66 1.71 2673.95 7051.
```
**Table 4**

Multi-objective multi-dimensional integer knapsack instances.

_p n m_ Algorithm | _N_ ( _Z_ ) | σ _N_ ( _Z_ ) _iter_ / _z_ ˆ _CPU_ (sec) σ _CPU_

```
3 25 2 S-AUGM 167.07 269.14 1.93 30.13 66.
GPBA-B 167.07 269.14 1.89 28.72 63.
GPBA-C 167.07 269.14 1.89 28.79 62.
```
```
3 S-AUGM 292.57 302.10 1.94 50.94 58.
GPBA-B 292.57 302.10 1.88 28.72 54.
GPBA-C 292.57 302.10 1.88 28.79 53.
```
```
4 S-AUGM 285.50 232.03 1.93 53.51 44.
GPBA-B 285.50 232.03 1.92 49.55 44.
GPBA-C 285.50 232.03 1.92 49.66 44.
```
```
5 S-AUGM 238.30 169.77 1.94 42.30 32.
GPBA-B 238.30 169.77 1.94 40.04 31.
GPBA-C 238.30 169.77 1.94 40.08 31.
```
putational time than the other two algorithms, around 3% less

than _GPBA-C_ , it is also the least efficient method in terms of

the number of iteration needed to compute each non-dominated

criterion vector. This suggests that both _GPBA-B_ and _GPBA-C_

computational time performance could be bounded by code

efficiency.

_4.3.4. Multi-objective multi-dimensional integer knapsack instances_

To assess the impact of the increase in dimensions on the al-

gorithms’ performance on integer instances, the instances from

Section 4.3.2 were applied using integer type variables. Table 4

shows the results for this case.

Although the impact of the increase in dimension does not

appear to be significant, the same trend as in Table 3 can be

observed where _S-AUGM_ is less efficient than _GPBA-B_ and _GPBA-_

_C_. In this case, _GPBA-B_ and _GPBA-C_ outperform consistently _S-_

_AUGM_ , both in terms of number of iterations solved per non-

dominated criterion vector obtained and in terms of computa-

tional time. The latter indicator also suggests that _GPBA-B_ is more

efficient than _GPBA-C_. The worse performance of _S-AUGM_ in in-

teger instances, can probably be attributed to the -constraint

model formulation used on _S-AUGM_ , which is not using elastic

constraints as proposed by Ehrgott and Ryan (2003). However,

( Table 4 ) as the number of constraints increase, the performance

difference between _S-AUGM_ and the other algorithms appears to

diminish.

_4.3.5. Multi-objective general integer programming instances_

Instances for general integer problems were generated, setting

a uniform distribution ranging between 0 and 20 for the objective

function’s profits and the constraints’ coefficients. The instances

were generated for both three and four objectives with multiple

constraints and 10, 15 and 20 variables. The generation of objective

function’s profits started with seed numbers 47, 63, 728 and 11,

one for each objective function. The initial seed numbers used to

generate the constraints’ coefficients was 634, 17, 28, 626, 135, 34,

78, 55, 783, 945, 823, 362, 133, 92 and 41, one for each constraint

equation. Seed numbers were increased by 5 on each instance of

each type. Results on solving these instances using _S-AUGM_ , _GPBA-_

_B_ and _GPBA-C_ are shown in Table 5

Results on the three objective instances, show the same trend

as in Section 4.3.4 , in which _GPBA-B_ and _GPBA-C_ outperform _S-_

_AUGM_ , although not as significantly. The less significant difference

in performance between _S-AUGM_ and the remaining algorithms

may be attributed to these problems having more constraints than

the ones presented in the previous section, where the performance

gap was lower as the number of constraints increased. Finally,

_GPBA-B_ is consistently the best performing algorithm among the

three.

When looking at the four objective instances, the same trend

is not apparent. On the contrary, _S-AUGM_ performs better than

_GPBA-B_ and _GPBA-C_ in all sets of problems, both in terms of com-

putational time and in terms of number of iterations per non-

dominated criterion vector. _GPBA-B_ , for all sets of problems, re-

quires the most iterations per non-dominated criterion vector,

while _GPBA-C_ takes the most time.

_4.4. Experiments on representation_

To assess the quality of the proposed algorithms for the rep-

resentation problem, experiments were carried out on the binary

knapsack problem instances, Section 4.3.1 , varying the representa-

tion’s control parameters according to the instance’s Pareto front

range. As a pre-computing stage for each instance, the acceptable

uniformity level, δ _k_ , and the acceptable coverage level, γ _k_ , are de-

fined as percentage of the range of the objective with the narrower

range. The ranges are obtained using the results of Section 4.3..

The cardinality control parameter required by algorithm _GP BA_ − _C_

is the maximum between the cardinality obtained by the cover-

age representation and the uniformity representation. Tables 6 , 7

and 8 present the results for these experiments for the instances

with three objectives and 50, 75 and 100 variables, respectively.

The same results are also depicted in the charts in Appendix B ,

on Figs. B.7 , B.8 and B.9 , respectively. Apart from the already in-

troduced performance measures, the obtained representations are

evaluated in terms of the final obtained coverage error, ( _R_ ( _N_ )) ,

the uniformity level, ( _R_ ( _N_ )) , the cardinality, ( _R_ ( _N_ )) and the

percentage of criterion vectors from the whole Pareto front that are

not covered by the defined coverage level for that instance, % _Ncov_..

In this section, only the three proposed algorithms are compared


**Table 5**

General multi-objective integer linear instances.

_p n m_ Algorithm | _N_ ( _Z_ ) | σ _N_ ( _Z_ ) _iter_ / ˆ _z CPU_ (sec) σ _CPU_

```
3 10 5 S-AUGM 19.83 10.14 1.80 1.54 1.
GPBA-B 19.83 10.14 1.80 1.37 0.
GPBA-C 19.83 10.14 1.80 1.45 0.
```
```
15 10 S-AUGM 38.93 20.13 1.79 3.72 2.
GPBA-B 38.93 20.13 1.78 3.36 1.
GPBA-C 38.93 20.13 1.78 3.45 1.
```
```
20 15 S-AUGM 75.13 31.07 1.73 9.03 3.
GPBA-B 75.13 31.07 1.73 8.74 3.
GPBA-C 75.13 31.07 1.73 8.91 3.
```
```
4 10 5 S-AUGM 30.67 13.31 3.54 7.73 4.
GPBA-B 30.67 13.31 3.58 7.75 4.
GPBA-C 30.67 13.31 3.55 7.86 4.
```
```
15 10 S-AUGM 107.77 74.05 3.60 43.39 37.
GPBA-B 107.77 74.05 3.64 44.23 39.
GPBA-C 107.77 74.05 3.63 44.70 38.
```
```
20 15 S-AUGM 311.30 175.06 3.52 152.01 94.
GPBA-B 311.30 175.06 3.56 156.39 97.
GPBA-C 311.30 175.06 3.55 156.43 97.
```
**Table 6**

Representation for multi-objective 0–1 knapsack instances with three objectives and fifty items.

_p n_ γ _k_ /δ _k_ Algorithm | ( _R_ ( _N_ )) | | ( _R_ ( _N_ )) | | ( _R_ ( _N_ )) | | % _Ncov_. | _iter_ / _z_ ˆ^ _CPU_ (s)

```
3 50 30% GPBA-A 15.87 208.97 41.97 6.12 1.42 0.
GPBA-B 9.10 316.63 67.93 13.59 1.15 0.
GPBA-C 11.77 225.97 57.27 12.02 1.22 0.
```
```
20% GPBA-A 32.40 174.13 21.23 8.33 1.43 2.
GPBA-B 16.13 247.70 47.53 14.24 1.18 1.
GPBA-C 24.10 189.93 30.77 14.65 1.20 1.
```
```
15% GPBA-A 41.07 162.83 17.37 11.68 1.45 2.
GPBA-B 24.33 235.67 29.17 19.82 1.18 1.
GPBA-C 28.77 181.87 22.43 20.88 1.19 1.
```
```
10% GPBA-A 77.10 125.03 11.53 12.01 1.61 5.
GPBA-B 41.67 195.53 20.37 23.66 1.19 2.
GPBA-C 50.93 150.53 14.43 22.79 1.22 2.
```
```
5% GPBA-A 155.23 95.30 6.87 17.24 2.11 13.
GPBA-B 93.47 149.93 10.47 32.42 1.25 5.
GPBA-C 85.10 133.30 10.10 40.59 1.27 4.
```
```
4% GPBA-A 171.77 90.60 6.07 21.98 2.32 16.
GPBA-B 116.57 125.67 9.17 34.93 1.27 6.
GPBA-C 88.70 127.50 10.80 50.12 1.28 5.
```
```
3% GPBA-A 215.43 85.87 5.00 23.27 2.85 25.
GPBA-B 149.90 117.10 7.97 37.70 1.33 9.
GPBA-C 105.70 118.27 8.90 57.87 1.32 6.
```
```
2% GPBA-A 259.37 77.60 4.33 23.92 3.60 39.
GPBA-B 199.10 110.20 6.33 37.57 1.41 13.
GPBA-C 114.63 118.07 8.73 65.30 1.33 6.
```
```
1% GPBA-A 331.57 62.50 3.47 15.59 6.05 92.
GPBA-B 277.17 79.10 4.57 27.17 1.57 21.
GPBA-C 131.70 109.83 8.13 66.81 1.36 7.
```
since _S-AUGM_ was developed exclusively to compute the whole

Pareto front and not a representation of it.

On the one hand, in Tables 6, 7 and 8 , as well as in Figs. B.7,

B.8 and B.9 it is possible to observe that, apart from few ex-

ceptions, _GPBA-A_ presents the best results on coverage error and

_GPBA-B_ on uniformity level. However, except in cases of lower ac-

ceptable coverage error/uniformity level, both algorithms show the

worst performance in the other measure, i.e. _GPBA-A_ on unifor-

mity level and _GPBA-B_ on coverage error. On the other hand, al-

gorithm _GPBA-C_ appears to be an algorithm with a more balanced

performance. When lower values for the target cardinality are set,

_GPBA-C_ obtains a good coverage error, similar to _GPBA-A_ , while

not compromising uniformity as much. For higher values of tar-

get cardinality, _GPBA-C_ fails to reach them, compromising on cov-

erage. Nevertheless, the lower cardinality levels allow for a better

uniformity. Additionally, even with lower cardinality than _GPBA-B_ ,

in many situations _GPBA-C_ is able to reach a better coverage er-

ror (e.g. see in Table 6 the 4% row, and in Table 7 the 3% and


## Representation for multi-objective 0–1 knapsack instances with three objectives and seventy-five items.

###### GPBA-C 38.80 252.07 25.20 13.61 1.15 2.

###### GPBA-B 55.07 263.43 19.60 17.52 1.15 3.

###### GPBA-C 310.00 140.83 4.83 53.39 1.23 20.

Representation for multi-objective 0–1 knapsack instances with three objectives and one hundred items.

- Table
   - 3 75 30% GPBA-A 17.67 314.43 52.30 6.85 1.37 1. p n γ k /δ k Algorithm | ( R ( N )) | | ( R ( N )) | | ( R ( N )) | | % Ncov | iter / z ˆ CPU (s)
            - GPBA-B 10.43 402.67 86.10 10.33 1.17 0.
            - GPBA-C 13.00 336.80 73.73 11.06 1.23 0.
      - 20% GPBA-A 40.63 260.77 20.80 6.94 1.31 2.
            - GPBA-B 19.57 403.37 41.60 14.45 1.15 1.
            - GPBA-C 31.03 265.00 31.53 9.81 1.16 1.
      - 15% GPBA-A 51.57 247.50 15.43 10.56 1.30 3.
            - GPBA-B 29.93 307.03 31.43 15.46 1.15 1.
            - GPBA-C 38.80 252.07 25.20 13.61 1.15 2.
      - 10% GPBA-A 116.63 176.97 10.07 7.39 1.33 7.
            - GPBA-B 55.07 263.43 19.60 17.52 1.15 3.
            - GPBA-C 81.40 208.17 12.67 13.73 1.14 4.
      - 5% GPBA-A 288.97 137.33 4.90 10.97 1.56 22.
            - GPBA-B 149.33 190.20 9.73 23.02 1.15 8.
            - GPBA-C 174.10 168.40 8.23 22.45 1.17 10.
      - 4% GPBA-A 333.23 132.40 4.43 14.47 1.67 27.
            - GPBA-B 198.90 182.97 6.97 26.20 1.17 11.
            - GPBA-C 193.63 163.53 8.07 30.09 1.19 11.
      - 3% GPBA-A 497.73 112.27 3.17 15.59 1.92 47.
            - GPBA-B 281.23 149.53 6.13 30.90 1.20 17.
            - GPBA-C 269.53 145.37 5.53 35.78 1.22 16.
      - 2% GPBA-A 637.53 100.63 2.67 22.05 2.32 75.
            - GPBA-B 428.90 131.37 4.40 36.87 1.25 28.
            - GPBA-C 310.00 140.83 4.83 53.39 1.23 20.
      - 1% GPBA-A 977.70 76.97 2.03 23.08 3.64 195.
            - GPBA-B 741.07 107.60 2.87 37.43 1.39 58.
            - GPBA-C 404.50 131.40 3.87 66.87 1.30 28.
- Table
   - 3 100 30% GPBA-A 16.80 393.43 55.93 6.04 1.39 1. p n γ k /δ k Algorithm | ( R ( N )) | | ( R ( N )) | | ( R ( N )) | | % Ncov | iter / z ˆ CPU (s)
               - GPBA-B 9.80 564.00 118.43 12.08 1.17 0.
               - GPBA-C 13.10 457.87 72.77 12.44 1.22 1.
         - 20% GPBA-A 42.77 315.07 24.80 5.22 1.27 2.
               - GPBA-B 19.20 483.30 57.73 13.36 1.14 1.
               - GPBA-C 35.43 340.33 28.20 9.07 1.15 2.
         - 15% GPBA-A 51.60 297.30 19.57 8.45 1.26 3.
               - GPBA-B 29.97 420.97 38.80 14.70 1.14 2.
               - GPBA-C 39.07 332.03 29.90 14.34 1.15 2.
         - 10% GPBA-A 126.97 223.17 9.90 7.06 1.25 8.
               - GPBA-B 58.87 356.37 22.03 16.43 1.12 3.
               - GPBA-C 93.10 260.50 13.93 10.96 1.12 5.
         - 5% GPBA-A 352.90 161.10 5.07 8.87 1.36 27.
               - GPBA-B 167.50 240.13 10.27 19.83 1.12 11.
               - GPBA-C 232.87 197.87 7.77 15.96 1.13 15.
         - 4% GPBA-A 411.80 158.67 4.77 11.70 1.40 33.
               - GPBA-B 233.13 206.87 7.93 20.91 1.12 15.
               - GPBA-C 253.77 192.90 8.10 21.77 1.14 17.
         - 3% GPBA-A 687.63 136.83 3.40 11.10 1.53 62.
               - GPBA-B 349.17 191.07 6.03 24.17 1.13 24.
               - GPBA-C 425.83 166.93 4.57 22.26 1.17 30.
         - 2% GPBA-A 945.80 126.90 2.53 17.00 1.76 101.
               - GPBA-B 585.87 159.00 4.03 30.41 1.17 43.
               - GPBA-C 499.13 161.00 4.03 38.58 1.19 37.
         - 1% GPBA-A 1706.87 96.67 1.77 23.85 2.57 285.
               - GPBA-B 1197.53 121.40 2.63 38.70 1.27 100.
               - GPBA-C 759.87 140.60 3.23 60.58 1.25 60.


4% rows). Finally, although _GPBA-C_ never reaches coverage errors

as low as _GPBA-A_ , it is much more efficient, both in number of

models solved per non-dominated criterion vector in the repre-

sentation, and in computational time. Regarding the percentage

of criterion vectors from the Pareto front that are not covered by

the defined coverage level, % _Ncov_. , _GPBA-A_ obtains the best mean

values in all experiments and with relatively low errors. Never-

theless, both _GPBA-B_ and _GPBA-C_ perform similarly on this indi-

cator, both less satisfactorily than _GPBA-A_. Moreover, comparing

the performance of _GPBA-B_ and _GPBA-C_ in light of the aforemen-

tioned indicator two considerations can be made: (1) _GPBA-C_ while

having a better coverage error than _GPBA-B_ , in some cases leaves

many points not covered at the desired coverage error; and (2) the

fact that _GPBA-B_ has a relatively low number of points not cov-

ered at the desired coverage error when comparing to the cov-

erage error suggests that the reason for its underperformance in

terms of coverage error is that whole areas of the Pareto front are

undercovered.

_4.5. Final comments and remarks_

All the three algorithms were tested on experiments set for

computing the whole Pareto front and a representation of it. Re-

garding the computation of the whole Pareto front:

1. All the three algorithms outperformed the _AUGM-2_ algorithm.

The reason for this is that _AUGM-2_ only skips redundant so-

lution and applies the early exit mechanism in the inner-

most loop. Furthermore, the redundant solution skip strategy

is less sophisticated than the one applied on the other al-

gorithms since it does not consider all past solutions, but

just the last one. These factors imply that _AUGM-2_ requires

more models to be solved per non-dominated criterion vector

computed.

2. Algorithms _GPBA-B_ and _GPBA-C_ outperform _GPBA-A_ and _R-_

_AUGM_. _GPBA-A_ , due to the strategy used to determine the 

vector, does not have the early exit mechanism, lagging behind

the other algorithms in terms of number of models solved per

non-dominated criterion vector computed.

3. Algorithms _GPBA-B_ and _GPBA-C_ performance is comparable with

_S-AUGM_ , being amongst the best in the literature. This was to

be expected since the acceleration mechanisms are common

to the three algorithms. However, probably due to the model

structure used, _S-AUGM_ falls behind on integer knapsack in-

stances with a low number of constraints (less than four). But,

for general multi-objective integer instances, _S-AUGM_ appears

to be more efficient.

Regarding the representation of the Pareto front, the three al-

gorithms were not compared with others in literature since, to the

best of our knowledge, these are the first -constraint based al-

gorithms to target the representation problem for integer prob-

lems with more than two objectives. Hence, the analysis of the

proposed algorithms performance on the representation of Pareto

front showed that:

1. _GPBA-A_ was the algorithm with the best coverage error, while

having the worst uniformity level.

2. _GPBA-B_ had the best uniformity level, presenting the worst cov-

erage error.

3. Contrary to what is observed for _GPBA-A_ and _GPBA-B_ , _GPBA-_

_C_ behaviour changes depending on the target cardinality and

shows a behaviour balanced between _GPBA-A_ and _GPBA-B_ ’s per-

formances in all dimensions of the representation problem.

When lower cardinality targets are used, _GPBA-C_ appears to

privilege coverage. Nevertheless, when higher cardinality tar-

gets are set, _GPBA-C_ is unable to increase the cardinality of the

representation and compromises on coverage while improving

uniformity. This hybrid behaviour seems very convenient, since

_GPBA-C_ is more efficient than _GPBA-A_.

**5. Conclusions**

This work addresses the Pareto front representation problem,

contributing with methodologies to explore the feasible region of

the objective space. This is relevant for most multi-objective prob-

lems, since the evaluation of the whole Pareto front may be over-

whelming for the decision-maker, potentially hindering the deci-

sion process. A good representation of the Pareto front consists of

as few points as possible, covers all regions of the objective space

and spreads the points the most. To tackle this problem three algo-

rithms are presented: one aiming at coverage, a second at unifor-

mity and a third at cardinality. All algorithms are based on the -

constraint method and are insensitive to the quality of nadir point

estimation.

The algorithms were tested on the ability to efficiently ob-

tain the Pareto front under 240 binary knapsack instances, 150

integer knapsack instances and 180 general problem instances,

with three and four objectives. The algorithms that target unifor-

mity and cardinality are demonstrated to be very efficient and

among the best in literature. The algorithms were also tested on

the quality of the obtained representation. Although the cover-

age and uniformity algorithms showed good results on their tar-

geted measures, the cardinality algorithm for lower cardinalities

presents a coverage error similar to the coverage algorithm, whilst

not compromising much on uniformity and being much more

efficient.

As future work we intend to study the incorporation of these

representation strategies in interactive algorithms. Namely, we be-

lieve that the cardinality algorithm will provide a good basis in the

sense that, in the first stage a good coverage of the Pareto front can

be obtained and, at a later stage, as the DM’s preferences are re-

fined and the solution space more restricted, a representation of

the selected area can be obtained with less effort by the cover-

age algorithm. Finally, we would like to apply these representa-

tion algorithms in real multi-objective problems that would ben-

efit from being treated as such, without the computational com-

plexity of computing the whole Pareto front and the difficulty of

analysing it.

**Acknowledgments**

Mariana Mesquita-Cunha acknowledges the support by national

funds through FCT, under the research grant SFRH/BD/149441/2019.

José Rui Figueira acknowledges the support by national

funds through FCT, under DOME research project, PTDC/CCI-

COM/31198/2017. Ana Paula Barbosa-Póvoa and Mariana Mesquita-

Cunha acknowledge the support by national funds through FCT,

under the Data2Help research project, DSAIPA/AI/0044/2018, and

the project 1801P.00740 PTDC/EGE-OGE/28071/2017 - LISBOA-01-

0145-FEDER-028071. At last, this work is financed by national

funds through the FCT - Foundation for Science and Technology,

I.P., under the project UIDB/0 0 097/2020. The authors would also

like to thank Doctor Kostas Florios for the valuable comments and

suggestions.


**Appendix A. Figures of the illustrative example**

**Fig. A.4.** Illustrative example for the coverage representation, _GPBA-A_. The acceptable coverage error was defined as γ= 15 in each objective. Only the first iteration over

the outermost loop is depicted. The blue line represents the  2 bound imposed over objective _z_ 2 , and the red line represents the  3 bound imposed over objective _z_ 3.


**Fig. A.5.** Illustrative example for the uniformity representation, _GPBA-B_. The uniformity level defined was δ= 10 in all objectives. Only the first iteration over the outermost

loop is depicted. The blue line represents the  2 bound imposed over objective _z_ 2 , and the red line represents the  3 bound imposed over objective _z_ 3.


**Fig. A.6.** Illustrative example for the cardinality representation, _GPBA-C_. The desired cardinality defined was 5 in all objectives. Only the first iteration over the outermost

loop is depicted. The blue line represents the  2 bound imposed over objective _z_ 2 , and the red line represents the  3 bound imposed over objective _z_ 3.


**Appendix B. Visualization of the results of experiments on**

**representation**

**Fig. B.7.** Impact of the control parameters’ restriction on the representation problem’s objectives for the proposed algorithms, when applying to the binary knapsack problem

instances in Section 4.3.1 with fifty variables.

**Fig. B.8.** Impact of the control parameters’ restriction on the representation problem’s objectives for the proposed algorithms, when applying to the binary knapsack problem

instances in Section 4.3.1 with seventy-five variables.

**Fig. B.9.** Impact of the control parameters’ restriction on the representation problem’s objectives for the proposed algorithms, when applying to the binary knapsack problem

instances in Section 4.3.1 with one hundred variables.


**References**

Alves, M. J., & Clímaco, J. (2007). A review of interactive methods for multiobjec-

```
tive integer and mixed-integer programming. European Journal of Operational
Research, 180 (1), 99–115. https://doi.org/10.1016/j.ejor.2006.02.033.
```
Alves, M. J., & Costa, J. P. (2009). An exact method for computing the nadir values in

```
multiple objective linear programming. European Journal of Operational Research,
198 (2), 637–646. https://doi.org/10.1016/j.ejor.2008.10.003.
```
Audet, C., Bigeon, J., Cartier, D., Le Digabel, S., & Salomon, L. (2021). Performance in-

```
dicators in multiobjective optimization. European Journal of Operational Research,
292 (2), 397–422. https://doi.org/10.1016/j.ejor.2020.11.016.
```
Branke, J., Deb, K., Miettinen, K., & Slowi nski, ́ R. (2008). _Multiobjective optimization:_

```
Interactive and evolutionary approaches. Berlin, Germany: Springer-Verlag. https:
//doi.org/10.1007/978- 3- 540- 88908- 3.
```
Ceyhan, G., Köksalan, M., & Lokman, B. (2019). Finding a representative nondomi-

```
nated set for multi-objective mixed integer programs. European Journal of Oper-
ational Research, 272 (1), 61–77. https://doi.org/10.1016/j.ejor.2018.06.012.
```
Chalmet, L., Lemonidis, L., & Elzinga, D. (1986). An algorithm for the bi-criterion

```
integer programming problem. European Journal of Operational Research, 25 (2),
292–300. https://doi.org/10.1016/0377- 2217(86)90093- 7.
```
Chankong, V., & Haines, Y. Y. (2008). _Multiobjective decision making: Theory and_

```
methodology. Mineola, NY, USA: Dover Publications Inc..
```
Coffin, M., & Saltzman, M. J. (20 0 0). Statistical analysis of computational tests of

```
algorithms and heuristics. INFORMS Journal on Computing, 12 (1), 24–44. https:
//doi.org/10.1287/ijoc.12.1.24.11899.
```
Do ̆gan, I., Lokman, B., & Köksalan, M. (2022). Representing the nondominated set

```
in multi-objective mixed-integer programs. European Journal of Operational Re-
search, 296 (3), 804–818. https://doi.org/10.1016/j.ejor.2021.04.005.
```
Dächert, K., Klamroth, K., Lacour, R., & Vanderpooten, D. (2017). Efficient compu-

tation of the search region in multi-objective optimization. _European Journal
of Operational Research, 260_ (3), 841–855. https://doi.org/10.1016/j.ejor.2016.05.
029.
Ehrgott, M. (2005). _Multicriteria optimization_ ((2nd ed.)). Berlin, Germany: Springer-

```
Verlag. https://doi.org/10.1007/3- 540- 27659- 9.
```
Ehrgott, M., & Ruzika, S. (2008). Improved ε-constraint method for multiobjective

```
programming. Journal of Optimization Theory and Applications, 138 (3), 375. https:
//doi.org/10.1007/s10957- 008- 9394- 2.
```
Ehrgott, M., & Ryan, D. M. (2003). The method of elastic constraints for multiob-

```
jective combinatorial optimization and its application in airline crew schedul-
ing. In T. Tanino, T. Tanaka, & M. Inuiguchi (Eds.), Multi-objective program-
ming and goal programming: vol. 21 (pp. 117–122). Berlin, Germany: Springer.
https://doi.org/10.1007/978- 3- 540- 36510- 5 _ 14.
```
Eusébio, A., Figueira, J., & Ehrgott, M. (2014). On finding representative non-

```
dominated points for bi-objective integer network flow problems. Computers &
Operations Research, 48 , 1–10. https://doi.org/10.1016/j.cor.2014.02.009.
```
Faulkenberg, S. L., & Wiecek, M. M. (2010). On the quality of discrete representa-

```
tions in multiple objective programming. Optimization and Engineering, 11 (3),
423–440. https://doi.org/10.1007/s11081- 009- 9099- x.
```
Haimes, Y., Lasdon, L., & Wismer, D. (1971). On a bicriterion formulation of the prob-

```
lems of integrated system identification and system optimization. IEEE Trans-
actions on Systems, Man, and Cybernetics, SMC-1 (3), 296–297. https://doi.org/10.
1109/TSMC.1971.4308298.
```
Isermann, H., & Steuer, R. E. (1988). Computational experience concerning payoff

```
tables and minimum criterion values over the efficient set. European Journal of
Operational Research, 33 (1), 91–97. https://doi.org/10.1016/0377-2217(88)90257-
3.
```
Kidd, M. P., Lusby, R., & Larsen, J. (2020). Equidistant representations: Connecting

```
coverage and uniformity in discrete biobjective optimization. Computers & Op-
erations Research, 117 , 104872. https://doi.org/10.1016/j.cor.2019.104872.
```
```
Kirlik, G., & Sayın, S. (2015). Computing the nadir point for multiobjective discrete
optimization problems. Journal of Global Optimization, 62 (1), 79–99. https://doi.
org/10.1007/s10898- 014- 0227- 6.
Kirlik, G., & Sayın, S. (2014). A new algorithm for generating all nondominated solu-
tions of multiobjective discrete optimization problems. European Journal of Op-
erational Research, 232 (3), 479–488. https://doi.org/10.1016/j.ejor.2013.08.001.
Klamroth, K., Lacour, R., & Vanderpooten, D. (2015). On the representation of the
search region in multi-objective optimization. European Journal of Operational
Research, 245 (3), 767–778. https://doi.org/10.1016/j.ejor.2015.03.031.
Klein, D., & Hannan, E. (1982). An algorithm for the multiple objective integer linear
programming problem. European Journal of Operational Research, 9 (4), 378–385.
https://doi.org/10.1016/0377-2217(82)90182-5.
Klingman, D., Napier, A., & Stutz, J. (1974). NETGEN: A program for generating large
scale capacitated assignment, transportation, and minimum cost flow network
problems. Management Science, 20 (5), 814–821. https://doi.org/10.1287/mnsc.20.
5.814.
Laumanns, M., Thiele, L., & Zitzler, E. (2006). An efficient, adaptive parameter vari-
ation scheme for metaheuristics based on the epsilon-constraint method. Euro-
pean Journal of Operational Research, 169 (3), 932–942. https://doi.org/10.1016/j.
ejor.2004.08.029.
Masin, M., & Bukchin, Y. (2007). Diversity maximization approach for multiobjective
optimization. Operations Research, 56 (2), 411–424. https://doi.org/10.1287/opre.
1070.0413.
Mavrotas, G. (2009). Effective implementation of the -constraint method in multi-
objective mathematical programming problems. Applied Mathematics and Com-
putation, 213 (2), 455–465. https://doi.org/10.1016/j.amc.2009.03.037.
Mavrotas, G., & Florios, K. (2013). An improved version of the augmented -
constraint method (AUGMECON2) for finding the exact Pareto set in multi-
objective integer programming problems. Applied Mathematics and Computation,
219 (18), 9652–9669. https://doi.org/10.1016/j.amc.2013.03.002.
Nikas, A., Fountoulakis, A., Forouli, A., & Doukas, H. (2020). A robust augmented
ε-constraint method (AUGMECON-R) for finding exact solutions of multi-
objective linear programming problems. Operational Research. https://doi.org/10.
1007/s12351- 020- 00574- 6.
Özlen, M., & Azizo glu, ̆ M. (2009). Multi-objective integer programming: A general
approach for generating all non-dominated solutions. European Journal of Oper-
ational Research, 199 (1), 25–35. https://doi.org/10.1016/j.ejor.2008.10.023.
Ozlen, M., Burton, B. A., & MacRae, C. A. G. (2014). Multi-objective integer program-
ming: An improved recursive algorithm. Journal of Optimization Theory and Ap-
plications, 160 (2), 470–482. https://doi.org/10.1007/s10957- 013- 0364- y.
Sayın, S. (20 0 0). Measuring the quality of discrete representations of efficient sets
in multiple objective mathematical programming. Mathematical Programming,
87 (3), 543–560. https://doi.org/10.1007/s101070 050 011.
Shao, L., & Ehrgott, M. (2016). Discrete representation of non-dominated sets in
multi-objective linear programming. European Journal of Operational Research,
255 (3), 687–698. https://doi.org/10.1016/j.ejor.2016.05.001.
Steuer, R. E. (1986). Multiple criteria optimization: Theory, computation and appli-
cation. New York, NY, USA: John Wiley & Sons. https://doi.org/10.1002/oca.
4660100109.
Sylva, J., & Crema, A. (2004). A method for finding the set of non-dominated vectors
for multiple objective integer linear programs. European Journal of Operational
Research, 158 (1), 46–55. https://doi.org/10.1016/S0377- 2217(03)00255- 8.
Sylva, J., & Crema, A. (2007). A method for finding well-dispersed subsets of non-
dominated vectors for multiple objective mixed integer linear programs. Euro-
pean Journal of Operational Research, 180 (3), 1011–1027. https://doi.org/10.1016/j.
ejor.2006.02.049.
Zhang, W., & Reimann, M. (2014). A simple augmented -constraint method for
multi-objective mathematical integer programming problems. European Journal
of Operational Research, 234 (1), 15–24. https://doi.org/10.1016/j.ejor.2013.09.001.
```

