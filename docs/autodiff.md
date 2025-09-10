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
    inputs: Sequence[easier.Tensor],
    outputs: Sequence[easier.Tensor],
) -> Tuple[
    Sequence[easier.Module],
    Sequence[easier.Tensor],
    Sequence[easier.Tensor]
]: ...
```

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
-   New `easier.Module`s:
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
def easier.jacobian(
    modules: Sequence[easier.Module],
    inputs: Sequence[easier.Tensor],
    outputs: Sequence[easier.Tensor],
) -> linsys.LinSys: ... # ?????
```

**TODO**:
-   it seems impossible for users to inspect the S/R.idx in LinSys,
    because it's somehow compressed,
    the idx will be somehow on concat-ed TensorGroups whose layout is EASIER-internal.

## References

1.  PyTorch AD APIs (graph style and functional style):
    https://docs.pytorch.org/docs/stable/autograd.html  

1.  PyTorch AD APIs (functional style):
    https://docs.pytorch.org/docs/stable/func.api.html