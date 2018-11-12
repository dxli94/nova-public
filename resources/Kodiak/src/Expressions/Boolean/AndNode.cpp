#include "AndNode.hpp"

using namespace kodiak;
using namespace kodiak::BooleanExpressions;

Certainty AndNode::doEvaluate(const Environment &environment, const bool useBernstein, const real eps) const {
    Certainty leftCertainty = this->lhs->eval(environment, useBernstein, eps);
    Certainty rightCertainty = POSSIBLY;
    switch (leftCertainty) {
        case FALSE:
            return FALSE;
        case TRUE:
            rightCertainty = this->rhs->eval(environment, useBernstein, eps);
            return rightCertainty;
        case POSSIBLY:
            rightCertainty = this->rhs->eval(environment, useBernstein, eps);
            if (rightCertainty == FALSE)
                return FALSE;
            else
                return POSSIBLY;
        case TRUE_WITHIN_EPS:
            rightCertainty = this->rhs->eval(environment, useBernstein, eps);
            if (rightCertainty == FALSE)
                return FALSE;
            else if (rightCertainty == TRUE_WITHIN_EPS)
                return TRUE_WITHIN_EPS;
            else if (rightCertainty == TRUE)
                return TRUE_WITHIN_EPS;
            else return POSSIBLY;
        default:
            throw Growl("Kodiak (AndOperator::eval): case not implemented");
    }
}

void AndNode::doPrint(std::ostream &cout) const {
    this->lhs->print(cout);
    cout << " && ";
    this->rhs->print(cout);
}

bool AndNode::equals(const Node &other) const {
    if (typeid(*this) != typeid(other))
        return false;

    const AndNode &otherCasted = dynamic_cast<AndNode const &>(other);
    return *this->lhs == *otherCasted.lhs && *this->rhs == *otherCasted.rhs;
}

unique_ptr<kodiak::BooleanExpressions::Node> AndNode::doClone() const {
    return std::make_unique<AndNode>(*this->lhs->clone(), *this->rhs->clone());
}

namespace kodiak {
    namespace BooleanExpressions {

        std::unique_ptr<kodiak::BooleanExpressions::Node>
        operator&&(std::unique_ptr<Node> const &lhs, std::unique_ptr<Node> const &rhs) {
            return std::make_unique<AndNode>(*lhs, *rhs);
        }

        std::unique_ptr<kodiak::BooleanExpressions::Node>
        operator&&(Node const &lhs, Node const &rhs) {
            return std::make_unique<AndNode>(lhs, rhs);
        }
    }
}
