# Autodiff in EASIER

## Open questions

1.  Will AD in EASIER escalate to $nO << nI$ cases, like taking gradient for loss in DL training?

    Otherwise, even for $nO \simeq nI$, forward-mode AD suffices.

1.  EASIER aggregators get involved in AD

1.  Stop gradient/derivatives.

    > Optional to stop on aggregators, make JVP even more sparse.

1.  Ensure AD coworks with distribution.

1.  Ensure recursive AD for higher order derivatives works.

1.  Assess the use scenarios e.g. with Python-world control flow.

1.  Interconnectability to PyTorch AD APIs or the interconnectability with jacobian/derivative/graident from PyTorch AD.

## Related work

1.  `torch.autograd` is famous for its backward-mode AD APIs, e.g.:

    ```
    torch.autograd.grad(
        outputs: Sequence[torch.Tensor],
        inputs: Sequence[torch.Tensor],
        grad_outputs: Sequence[torch.Tensor] = None,
        retain_graph=None,
        create_graph=False,
        # many other parameters
    ) -> Sequence[torch.Tensor]
    ```
    where all `torch.Tensor`s are nodes in PyTorch's built-up-at-runtime
    computational graph.

    For forward-mode AD, `autograd` provides two styles:

    -   ```
        torch.autograd.forward_ad.make_dual(
            tensor: torch.Tensor,
            tangent: torch.Tensor
        ) -> _DualTensor
        ```
        for users to manually pack dual numbers;

    -   ```
        torch.autograd.functional.jvp(
            func,
            inputs: Sequence[torch.Tensor],
            v: torch.Tensor
        ) -> Tuple[
            Sequence[torch.Tensor],  # equal to func(inputs)
            Sequence[torch.Tensor]   # Jacobian-vector product
        ]
        ```
        which is more functional-style and concise.
    
    However, `autograd` APIs do not seem directly adaptable to EASIER,
    as `autograd` relies on value-based `torch.Tensor`.

1.  Solution to memory consumption during backward propagation:

    Checkpoint primal values and recompute during backprop, e.g.

    ```
    o------>o------>o------>o------>o
                            x<------x
                    o------>o
                    x<------x
            o------>o
            x<------x
    o------>o
    x<------x
    ```

    where point `o` means primal checkpoint, `o-->` means computation of primal
    values, and `x<--x` means backprop between checkpoints.

1.  Sparsity of Jacobian matrix:

    "Non-interleaving" basis vectors
    (evaluation of JVP on basis vectors form columns of the Jacobian matrix)
    can be colored and grouped, so that a linear combination of basis vectors
    can be processed in a simultaneous manner.

1.  _Duality_ between pushforward and pullback.

## EASIER AD APIs

### Forward-mode (directional derivative)

```python
def easier.jvp(
    modules: Sequence[easier.Module],
    inputs: Sequence[easier.Tensor],   # x
    outputs: Sequence[easier.Tensor],  # y
) -> Tuple[
    Sequence[easier.Module],  # jvp_m
    Sequence[easier.Tensor],  # tangents_x
    Sequence[easier.Tensor]   # tangents_y
]: ...
```

**TODO**:
-   Aren't esr.Modules returned by `easier.jvp` already `_Pushforward`s?

-   When `tangent_x: easier.Tensor` is distributed, how can user set its value?

    NOTE if like `jax.jvp` and treat `tangent_x` as value-immediately-ready,
    this API becomes traditional `jvp` and we'll inevitably provide `jacobian` API.

-   Should the input/output Modules not be a collection, but a single Module?
    I.e. can different Modules run in arbitrary order, perhaps interleavedly?

    If we take only one Module,
    we may encapsulate the related descriptive `easier.Tensor`s into the class:
    ```python
    class _Pushforward(easier.Module):
        tangents_x: Sequence[easier.Tensor]
        tangents_y: Sequence[easier.Tensor]
        ...

    def easier.jvp(module: easier.Module, inputs, outputs):
        class _PushforwardInstance(_Pushforward):
            # Instantiate the class as `forward()` method is bound to class
            ...
        return _PushforwardInstance(m)
    
    pushforward = easier.jvp(m, [m.x], [m.y])
    [tg_x] = pushforward.tangents_x  # impossible to mix positions in tuple
    ```
    This may make the API code / user code more self-documentary and
    less error-prone, especially dealing with "tangent" "cotangent"
    in the same system.

-   It seems not suitable to call it `jvp` anymore,
    since we don't evaluate the _product_ of Jacobian and vector immediately,
    or even have (the value of) $v$ immediately.

    Alternative names:
    1.  `jacobian`:
        This API returns actually a subprocedure equivalent to Jacobian matrix.
        But the problem may be we don't have a term in this way for "vjp"
        -- `jacobian_tranpose`? (might it emphasize too much the nature of being matrix while it's not?)

    1.  `pushforward`:
        Too uncommon? But the dual API could be `pullback`.

    1.  `tangent_map` and `cotangent_map`
    1.  `forward_map` and `backward_map`

    1.  `derivative/differential` and `adjoint`: in a sense of e.g. "differential operator" and "adjoint operator"

    Remarkably, `jacfwd/jacrev` APIs in JAX, `torch.func` etc. do not reflect
    the duality here. They both calculate the Jacobian (pushforward) only
    but in different ways.


Represents:
-   Jacobian-vector product

-   _Pushforward_ in
    $$
    \bigoplus_i \left( T_{(X_i)} \mathbb{R}^{S_i} \right)
    \to
    \bigoplus_i \left( T_{(Y_j)} \mathbb{R}^{U_j} \right)
    $$

    where $X_i$, $Y_j$ mean values of `inputs[i]`, `outputs[j]`,
    and $S_i$, $U_j$ mean shapes of `inputs[i]`, `outputs[j]`,
    and $T_x M$ means the tangent space of space $M$ at $x\in M$
    (especially, $T_x\mathbb{R^n}$ is isomorphic to $\mathbb{R}^n$).


Arguments:
-   `inputs/outputs`: `easier.Tensor` included in `modules`

Returns:
-   New `easier.Module`s (`easier._Pushforward`s ??):
    -   Inherit all original `easier.Tensor`s
    -   After execution, all original `easier.Tensor`s are filled with _primal results_
    -   After execution, new output tangent  `easier.Tensor`s are filled Jacobian-vector product results
-   New input tangent `easier.Tensor`, whose values are read when any resultant `easier.Module` is executed
-   New output tangent `easier.Tensor`, which are written when any resultant `easier.Module` is executed

    > All tangents are read/written, maybe solely for initialization.

Usage:
```python
m = Module()

[jvp_m], [tg_x], [tg_y] = easier.jvp([m], [m.x], [m.y])
assert jvp_m is not m

# Use tangent easier.Tensor in other easier.Modules
m2 = AnotherModule(tg_y)

[jvp_m] = easier.compile([jvp_m], backend='torch')

for i in range(10):
    jvp_m()

    # Use ry, ty with outside PyTorch tangent `torch.Tensor`s
    ry = m.y.collect()
    ty = tg_y.collect()
    # TODO how to fill distributed `m.x, tg_x: easier.Tensor` if there is
    # outside computation that is NOT replicated easier.Module?
```


> This is essentially another representation of Jacobian matrix:
> - resultant Modules encode the linear map of _matmul with Jacobian matrix_
> - Jacobian matrix is likely sparse

> **TODO**
> - How to explicitly encode $\oplus_i T_{X_i} \mathbb{R}^{S_i}$
>   above into domain of the linear map?
> - How to compress the `.idx` (maybe data, too) for computational structure
>   into nested DataLoaders?

> Since all easier.Tensors in module do have concrete values to begin with,
> we can extraly select a subset of `get_easier_objects(modules)`, and other easier.Tensors excepts `inputs @ outputs` become
> _free variables_ -- when we switching from description to arbitrary evaluation,
> values of those free variables may _change_.

> easier.Tensor has ctor parameter `require_grad`, emphasizing _gradients_ which are not in this case,
> we could only treat them as the real-and-only inputs, also ouputs.

### Forward-mode (Jacobian matrix)
```python
class _Pushforward(easier.Module):
    ...

    # def dual(self) -> _Pullback: ...
    # TODO where to get cot_y, cot_x for the resultant pullback?
    # users can simply call easier.vjp, passing the same arguments?

def easier.jacobian(
    modules: Sequence[easier.Module],
    inputs: Sequence[easier.Tensor],   # x
    outputs: Sequence[easier.Tensor],  # y
) -> Tuple[
    Sequence[_Pushforward],   # pushforward
    Sequence[easier.Tensor],  # tangent_x
    Sequence[easier.Tensor]   # tangent_y
]: ...
```


#### Open question: About sparsity in Jacobian and linear combination of direction bases

Such properties may be leverage to calculate gradient _using forward AD_, where
backward AD is theoretically better given $nI >> nO = 1$ but comes at the cost
of memory consumption.

For example, given
$f\in \mathbb{R}^n \to \mathbb{R}$,
the pushforward at $x\in\mathbb{R}^n$ is
$df_x\in \mathbb{R}^n \to \mathbb{R}$ too (up to isomorphism).
We may need to call $df_x(e_i)$ for all bases $e_i$ to get the components of
gradient vector.

However, if Jacobian matrix has sparsity (an extreme case is map-then-sum),
we may have an operator $\mathcal{V}$ to convert (e.g. "vmap")
$df_x\in \mathbb{R}^n \to \mathbb{R}$
to
$\mathcal{V}(df_x) \in \mathbb{R}^n \to \mathbb{R}^n$.

Then we can call $\mathcal{V}(df_x)(\sum_i e_i)$ of some proper linear combination.

And this may be extended to general sparsity (e.g. `Selector/Reducer`) and
general pullback.

Open questions:

-   The algorithm correctness
-   Given a `_Pushforward`, `easier.jvp/jacobian` returns
    a single (sequence of) tangent `easier.Tensor`, it may not be flexible
    enough to carry the linear combination of bases given the arbitrariness
    of the Jacobian sparsity.

-   We may provide `_Pushforward.dual(): _Pullback` only and specifically to
    serve as $\mathcal{V}$, generalized for general pullbacks,
    and such a pullback is numerically equivalent to pullback from `easier.vjp`
    but implementation-wise different: using forward AD v.s. backward AD.
    


### Backward-mode
```python
class _Pullback(easier.Module):
    ...

#     def dual(self) -> _Pushforward: ...
# [pushforward], [tg_x], [tg_y] = easier.jvp()
# pullback = pushforward.dual()
# TODO where to get cot_y, cot_x ?

def easier.vjp(
    modules: Sequence[easier.Module],
    inputs: Sequence[easier.Tensor],   # x
    outputs: Sequence[easier.Tensor],  # y
) -> Tuple[
    Sequence[easier.Module],  # vjp_m  : _Pullback???
    Sequence[easier.Tensor],  # cotangent_y
    Sequence[easier.Tensor]   # cotangent_x
]: ...

def easier.grad(
    modules: Sequence[easier.Module],
    inputs: Sequence[easier.Tensor],  # x
    output: easier.Tensor,            # y
) -> Tuple[
    Sequence[easier.Module],  # grad_m
    Sequence[easier.Tensor],  # grad_x
]: ...
```


## References

1.  PyTorch AD APIs (graph style and functional style):
    https://docs.pytorch.org/docs/stable/autograd.html  

1.  PyTorch AD APIs (functional style):
    https://docs.pytorch.org/docs/stable/func.api.html