# Autodiff in EASIER

## Open questions

1.  Will AD in EASIER escalate to $\#O << \#I$ cases, like taking gradient for loss in DL training?

    Otherwise, even for $\#O \simeq \#I$, forward-mode AD suffices.

1.  EASIER aggregators get involved in AD

1.  Stop gradient/derivatives.

1.  Ensure AD coworks with distribution.

1.  Assess compatibility with Hessian etc.

1.  Assess the use scenarios e.g. with Python-world control flow.

1.  Interconnectability to PyTorch AD APIs or the interconnectability with jacobian/derivative/graident from PyTorch AD.

## Design principles

1.  Given the descriptive nature of `easier.Module` and `easier.Tensor` (before entering `easier.compile`),
    EASIER AD style is more like the graph-style `autograd` APIs of PyTorch.

    `autograd` is famous for its backward-mode AD APIs, e.g.:
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
    where all `torch.Tensor`s are nodes in PyTorch internal computational graph.

    For forward-mode AD, `autograd` provides two styles:

    -   ```
        torch.autograd.forward_ad.make_dual(
            tensor: torch.Tensor, tangent: torch.Tensor
        ) -> _DualTensor
        ```
        for users to manually pack dual numbers;

    -   ```
        torch.autograd.functional.jvp(
            func, inputs: Sequence[torch.Tensor]
        ) -> Tuple[
            Sequence[torch.Tensor],  # equal to func(inputs)
            Sequence[torch.Tensor]   # Jacobian-vector product
        ]
        ```
        which is more functional-style and concise.
    
    However, both ways for forward-mode AD do not seem directly adaptable to EASIER primitives.

## EASIER AD APIs

### Forward-mode {directional derivative}???

```
easier.jvp(
    module: easier.Module,
    formal_inputs: Sequence[easier.Tensor]
) -> ?????
```

> Since all easier.Tensors in module do have concrete values to begin with,
> we can extraly select a subset of formal_inputs, and other easier.Tensors become
> _free variables_ -- when we switching from description to arbitrary evaluation,
> values of those free variables may _change_.

> Another way to see this is we first do an inclusion $\iota: E \hookrightarrow T_x R^{shp1++...++shpN}$
> then we get $Df \circ \iota: E \to T_y R^{shp1++...++shpN}$

> ~~Tangent space on $y$ is isomorphic to tangent space on $x$? Probably because we haven't strictly
> divided esr.Tensors into domain/codomain in a math function way?~~
>
> easier.Tensor has ctor parameter `require_grad`, we could only treat them as the real-and-only inputs, also ouputs.

### Forward-mode Jacobian
```
easier.jacobian(
    module: easier.Module,
    inputs: Sequence[easier.Tensor]
) -> ?????
```

Arguments:
-   `module`
-   `inputs`: a subset of `easier.Tensor`s contained by `module`

## References

1.  PyTorch AD APIs (graph style and functional style):
    https://docs.pytorch.org/docs/stable/autograd.html  

1.  PyTorch AD APIs (functional style):
    https://docs.pytorch.org/docs/stable/func.api.html