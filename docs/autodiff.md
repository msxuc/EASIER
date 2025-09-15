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

1. JAX AD: `jvp` and `jacfwd`

    JAX AD APIs are all dealing with functionals (function definitions in JAX IR),
    therefore recursively application of JAX AD APIs is essentially function composition.
    The function definition will be (JAX-JIT-compiled and) evaluated when it meets input arrays for the first time.

    `jax.jacfwd` $\in (R^n \to R^m) \to R^n \to R^{m\times n}$
    and internally it calls `jax.jvp`:

    ```python
    def jacfwd(fun):
        def jacfun(x):
            # jvp returns (primals, tangents)
            pushfwd = lambda tg_x: jax.jvp(fun, x, tg_x)
            y, J = jax.vmap(
                pushfwd,
                # dont vmap primals;
                # vmap along 2nd dim of tangents (n in mxn)
                out_axes=(None, -1)
            )(jax.eye(x.size))
            return J
        return jacfun

    x: jax.Array
    J: jax.Array = jacfwd(f)(x)
    ```

    > As an implementation detail, although the call to `jax.jvp` is delayed, encapsulated in
    a lambda function and passed in `jax.vmap` -- another JAX transformation of functional,
    the AD transformation in `jax.jvp` will still be fully realized before the
    vectorization transformation.
    >
    > So, `jacfwd` can be seen purely as an additional transformation to `jvp`.
    >
    > And the implementation of `jax.vmap` is not very relevant, as JAX vectorization is designed
    to respect operator-level vectorization rule, instead of the data sparsity
    like we can tell from a CSR matrix instance.

    `jax.jvp` evaluates a single pushed-forward tangent vector.
    If the input tangent vector is basis vector of the vector space for `x`,
    i.e. for **one scalar in `x`**, the result tangent vector will be a **column**
    in the corresponding Jacobian matrix.

    `jax.jacfwd` evaluates the whole Jacobian matrix by equivalently evaluates
    all columns. However, instead of doing a Python loop
    `for i in range(x.size): jax.jvp(fun, x, jax.eye(x.size)[i, :])`,
    it utilizes `jax.vmap` to vectorize the push forward functional first, so that all
    basis vectors -- the `eye(x.size)` -- can be pushed forward in a batch.

    `jax.jvp` does the AD transformation using _tracer_ approach, similar to
    `torch.fx` traces the `torch.nn.Module`:

    ```python
    x: jax.Array
    tg_x: jax.Array

    trace = jax.JVPTrace()  # similar to fx.Tracer
    in_tracer = jax.JVPTracer(trace, x, tg_x)  # similar to fx.Proxy
    ans = original_fun(in_tracer)

    # For the sake of simplicity, the code snippet assumes primitive to have a single parameter
    class JVPTrace:
        def process_primitive(self, primitive, tracers):
            primals_in = [tracer.primal for tracer in tracers] 
            tangents_in = [tracer.tangent for tracer in tracers] 

            # primitive_jvps is a global registry for JAX operators
            jvp = primitive_jvps.get(primitive)
            # type: (List[jax.Array], List[jax.Array]) -> (List[jax.Array], List[jax.Array])

            with jax.core.set_current_trace(self.parent_trace):
                # Normally, switch to EvalTrace
                primals_out, tangents_out = jvp(primals_in, tangents_in)

            return [
                JVPTracer(primal_out, tangent_out) for primal_out, tangent_out
                in zip(primals_out, tangents_out)
            ]
    
    # For JAX operators whose differential rules are highly dataflow-like:
    mul_prim: jax.Primitive
    jax.ad.defjvp(
        mul_prim,
        lambda xdot, x, y: jax.mul(xdot, y),
        lambda ydot, x, y: jax.mul(x, ydot),
    )
    # What API jax.mul looks like:
    def jax.mul(x, y):
        # When without special Trace set, use EvalTrace to evalute values.
        return mul_prim.bind(x, y)

    def defjvp(primitive, *jvprules):
        def jvp(primals, tangents):

            # Primitive itself is for hooking into the tracing process,
            # Primitive.bind() will evaluate using the Trace in the context.
            #
            # The `set_current_trace(self.parent_trace)` in JVPTrace has
            # switched this evaluation to use EvalTrace -- the tangent values
            # will be calculated on the fly.
            val_out = primitive.bind(*primals)
            tangents_out = [
                rule(tg, *primals) for rule, tg
                in zip(jvprules, tangents)
            ]
            return val_out, functools.reduce(jax.add, tangents_out)

        primitive_jvps[primitive] = jvp  
    ```


1.  https://github.com/mfschubert/sparsejac/blob/main/src/sparsejac/sparsejac.py

    Basically replace the `jax.eye(x.size)` above to a matrix whose rows are
    linear combinations of basis vectors of the tangent space for `x`,
    leveraging the predefined sparsity of Jacobian matrix.

    The algorithm:

    1. Jacbobian sparsity

        Consider all input scalars are vertices of a graph, and the target
        function to be a composition of many subfunctions.
        Any variables being arguments to the same subfunction are connected
        in the graph.
        The connectivity is propagated, e.g. `f2(f1(x, y), z)` leads to edges
        `x-y, y-z, z-x`.

        (in the above code, the sparsity is given by a parameter BCOO matrix).

    1.  Color the graph so that no adjacent vertices have the same color
        and minimize the number of colors $C$.

    1.  For each color, we can add basis tangent vectors of those input scalars
        and call `jvp` with that linear combination tangent vector.

        But for all colors, we still need to call `jvp` $C$ times.

    For EASIER:

    1.  Track the connectivity between input scalars through all intermediate
        results in the target `easier.Module`, so that each element in
        the intermediate tensor carries a union of IDs of connected input scalars

        > Reamrkably, it's each intermediate **tensor**,
        > not each intermediate TensorGroup.

        regarding:

        -   mapped operators: union remains unchanged

        -   Selector: copy unions

        -   Reducer: union unions that are reduced into the same output element

        And finally for each union of IDs, and for every two IDs in that union,
        add an edge to the graph for coloring.

1.  Hyper-dual number

    A theoretical framework that extends the algebra for dual number and differential rules,
    so that more than one basis tangent vectors can be encoded.

    Given $\dot x_k \in \mathbb{R}^n$:
    $$
    \hat x = x + \sum_{k<N} \epsilon_k \dot x_k
    $$

    $$
    f(\hat x) = f(x) + \sum_{k<N} \epsilon_k \left( J_f(x) \dot x_k \right)
    $$

    Differential rule for multiplication as an example:

    $$
    \hat a \hat b = ab + \sum_{k<N} \epsilon_k \left(a \dot b_k+\dot a_k b \right)
    $$

    It shows the application of differential rule (multiplication of Jacobian matrix)
    is done in a batch manner on dual parts $\{\epsilon_k\}$.

    

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

1.  ~~Solution to memory consumption during backward propagation:~~

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

1.  ~~_Duality_ between pushforward and pullback~~.

## EASIER AD APIs

### JVP (proposal)

Resultant module is a composition of operators and primitives.

When run, it:
1.  reads latest values in `inputs, outputs, tangents_in` and other attribute
    `easier.Tensor`s in `modules`;
1.  evaluates both primals and tangents and writes them into
   `outputs, tangents_out`.

```python
def easier.jvp(
    module: easier.Module,
    inputs: Sequence[easier.Tensor],
    outputs: Sequence[easier.Tensor],
) -> Tuple[
    easier.Module,
    Sequence[easier.Tensor],
    Sequence[easier.Tensor]
]: ...
```

Usage:
```python
m = Module()

jvp_m, [tg_x], [tg_y] = easier.jvp(m, [m.x], [m.y])
assert jvp_m is not m

[
    jvp_m, write_tg_x, read_tg_y
] = easier.compile([
    jvp_m, write_tg_x, read_tg_y
], backend='torch')

for i in range(10):
    write_tg_x()

    jvp_m()

    read_tg_y()
```

### Jacobian matrix using forward AD (proposal)

Resultant module is basically an SpMV, its matrix elements are fixed and
not dynamically depending on `inputs, ouputs` or attribute `easier.Tensor`s.

(The resultant elements for the Jacobian matrix and their layout are not visible to users)

When run, it:
1.  reads latest values in `tangents_in`;
1.  does SpMV and writes them into `tangents_out`.

```python
class Jacobian(easier.Module):
    def transpose(self) -> _Jacobian: ...
    # TODO make tg_x tg_y attributes of _Jacobian?

def easier.jacfwd(
    modules: easier.Module,
    inputs: Sequence[easier.Tensor],
    outputs: Sequence[easier.Tensor],
) -> Tuple[
    easier.Module,  # Jacobian calculator
    easier.Jacobian,  # Jacobian SpMV
    Sequence[easier.Tensor],
    Sequence[easier.Tensor]
]: ...
```

Usage:
```python
m = Module()

jacfwd_m, jacobian_m, [tg_x], [tg_y] = easier.jacfwd(m, [m.x], [m.y])

[
    jacfwd_m, jacobian_m, write_tg_x, read_tg_y
] = easier.compile([
    jacfwd_m, jacobian_m, write_tg_x, read_tg_y
], backend='torch')

jacfwd_m()

for i in range(10):
    write_tg_x()

    jacobian_m()

    read_tg_y()
```

Open questions about `jacfwd`:
1.  To get the Jacobian matrix, we must see the values of inputs, with
    input-tangents-in-JVP-sense being `torch.eye(input_size)`.

    But after the calculation of Jacobian matrix, we don't need input values
    any more and can use the Jacobian matrix on its own,
    as long as algorithmically the input values remain unchanged.

    The above API explicitly splits the two stages.
    (calculation of Jacobian matrix and Jacobian SpMV on whatever tangent vector)
    This is possible as the sparsity structure of Jacobian matrix can be inferred
    from input `m: easier.Module`.

    Otherwise, if we want a single pass to get the SpMV module,
    it requires the calculation of Jacobian matrix to be done during the call to `easier.jacfwd` itself.
    Consequently, if input values are runtime values, the `easier.jacfwd` must work as
    `easier.compile() + easier.Module.forward()`.
    
    > Challenge: we may need to dynamically re-layout distributed tensors.



### Open questions

1.  Dense Jacobian matrix for optimization problems

    For optimization problems, the last step would involve reduction into
    low-dimensional results, causing the Jacobian matrix to be dense.

    Probably we can offer a backward Jacobian API so that the last step
    can be separatedly evaluated using backward propagation.

    ```python
    def easier.jacrev(...)
    ```

### ~~Forward-mode (directional derivative)~~

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

## References

1.  PyTorch AD APIs (graph style and functional style):
    https://docs.pytorch.org/docs/stable/autograd.html  

1.  PyTorch AD APIs (functional style):
    https://docs.pytorch.org/docs/stable/func.api.html