# AddExample

## Product Support

| Product | Supported |
| ---- | :----:|
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Function Description

- Operator function: implements addition computation.

- Computation formula:

$$
y = x1 + x2
$$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>Input parameter for add_example computation, x1 in the formula.</td>
      <td>FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input parameter for add_example computation, x2 in the formula.</td>
      <td>FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter for add_example computation, y in the formula.</td>
      <td>FLOAT, INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Precautions

None

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Invocation Sample</th>
    <th>Description</th>
  </tr></thead>
  <tbody>
    <tr>
      <td>aclnn invocation</td>
      <td><a href="./examples/test_aclnn_add_example.cpp">test_aclnn_add_example</a></td>
      <td rowspan="2">Refer to [Operator Invocation](../../docs/en/invocation/quick_op_invocation.md) to complete operator compilation and verification.</td>
    </tr>
    <tr>
      <td>Graph mode invocation</td>
      <td><a href="./examples/test_geir_add_example.cpp">test_geir_add_example</a></td>
    </tr>
  </tbody>
</table>
