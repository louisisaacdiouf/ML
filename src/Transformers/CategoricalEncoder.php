<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\RuntimeException;

use function count;
use function array_values;
use function array_merge;
use function array_fill;
use function array_flip;

/**
 * Categorical Encoder
 *
 * The Categorical Encoder takes a feature column of categorical values and replace all
 * values by a numerical category representation.
 * Categorical encoding is typically used to convert
 * categorical data to continuous so that it can be used to train a learner that is only
 * compatible with continuous features.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Louis Issac Jean Samba DIOUF
 */
class CategoricalEncoder implements Transformer, Stateful, Persistable
{
    use AutotrackRevisions;

    /**
     * The set of unique possible categories per feature column of the training set.
     *
     * @var array<int[]>|null
     */
    protected ?array $categories = null;

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return DataType::all();
    }

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return isset($this->categories);
    }

    /**
     * Return the categories computed during fitting indexed by feature column.
     *
     * @return array<string[]>|null
     */
    public function categories() : ?array
    {
        return isset($this->categories) ? array_map('array_flip', $this->categories) : null;
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $this->categories = [];

        foreach ($dataset->featureTypes() as $column => $type) {
            if ($type->isCategorical()) {
                $values = $dataset->feature($column);

                $categories = array_values(array_unique($values));

                /** @var int[] $offsets */
                $offsets = array_flip($categories);

                $this->categories[$column] = $offsets;
            }
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param list<list<mixed>> $samples
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if ($this->categories === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            $vectors = [];

            foreach ($this->categories as $column => $categories) {
                $category = $sample[$column];

                // $template = array_fill(0, count($categories), 0);

                if (isset($categories[$category])) {
                    // $template[$categories[$category]] = 1;
                    $sample[$column] = $categories[$category];
                }

                // $vectors[] = $template;

                // unset($sample[$column]);
            }

            // $sample = array_merge($sample, ...$vectors);
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Categorical Encoder';
    }
}
